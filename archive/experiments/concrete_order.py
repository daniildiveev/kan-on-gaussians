"""Prediction on held-out concrete mixtures from UCI dataset 165."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, product
import json
import math
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch import nn

from experiments import geometry_benchmark as base, hypergraph_order as scalar
from src.experiment import _digest, _runtime, _setup_torch, _write_json, rmse
from src.models import parameter_count

DATA = Path(__file__).resolve().parents[1] / "data" / "concrete_compressive_strength.csv"
OUTPUT = Path("outputs/study/concrete_order")
SOURCE = "https://archive.ics.uci.edu/static/public/165/data.csv"
COUNTS = {"II": 993, "III": 1001, "MLP": 998}
KERNELS = {"II": {1: 6, 2: 5}, "III": {1: 4, 2: 2, 3: 1}}
FOLDS, SPLIT_SEED = 5, 13601
DEV_INITS, FINAL_INITS = (83, 89), (101, 103, 107)
RATES, EPOCHS, PATIENCE, BATCH = (0.001, 0.003, 0.01), 180, 25, 64
HEADER = ("Cement,Blast Furnace Slag,Fly Ash,Water,Superplasticizer,Coarse Aggregate,"
          "Fine Aggregate,Age,Concrete compressive strength")


@lru_cache(maxsize=1)
def _dataset():
    with DATA.open() as stream:
        assert stream.readline().strip() == HEADER
        data = np.loadtxt(stream, delimiter=",")
    assert data.shape == (1030, 9) and np.isfinite(data).all() and (data[:, 7:] > 0).all()
    _, group = np.unique(data[:, :7], axis=0, return_inverse=True)
    x = data[:, :8].copy()
    x[:, -1] = np.log1p(x[:, -1])
    return x, data[:, -1], group


def _folds(group):
    permutation = np.random.default_rng(SPLIT_SEED).permutation(np.unique(group))
    blocks = np.array_split(permutation, FOLDS)
    folds = []
    for fold, test in enumerate(blocks):
        pool = np.concatenate([block for i, block in enumerate(blocks) if i != fold])
        rng = np.random.default_rng(np.random.SeedSequence([SPLIT_SEED, fold, 1]))
        validation, train = np.split(rng.permutation(pool), [round(0.2 * len(pool))])
        assert not set(train) & set(validation) and not set(train) & set(test)
        assert not set(validation) & set(test)
        folds.append({split: np.flatnonzero(np.isin(group, groups)).tolist()
                      for split, groups in (("train", train), ("val", validation), ("test", test))})
    assert sorted(i for fold in folds for i in fold["test"]) == list(range(len(group)))
    return folds


class ConcreteLayer(nn.Module):
    forward = scalar.OrderLayer.forward

    def __init__(self, name):
        super().__init__()
        self.affine = nn.Linear(8, 1, dtype=torch.float64)
        self.banks = nn.ModuleList((scalar.PairBank if n == 2 else scalar.GaussianBank)(
            tuple(combinations(range(8), n)), k) for n, k in KERNELS[name].items())


def make_model(name, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if name == "MLP":
            return nn.Sequential(nn.Linear(8, 27, dtype=torch.float64), nn.ReLU(),
                                 nn.Linear(27, 26, dtype=torch.float64), nn.ReLU(),
                                 nn.Linear(26, 1, dtype=torch.float64), nn.Flatten(start_dim=0))
        return ConcreteLayer(name)


def freeze_protocol(root=OUTPUT):
    _, _, groups = _dataset()
    protocol = {
        "source": SOURCE, "dataset_doi": "10.24432/C5PK67", "license": "CC BY 4.0",
        "dataset_sha256": sha256(DATA.read_bytes()).hexdigest(), "rows": len(groups),
        "group_count": len(np.unique(groups)), "group_rule": "exact tuple of seven ingredients, excluding age",
        "split_seed": SPLIT_SEED, "folds": _folds(groups), "input_transform": "log1p(age), train-only standardization",
        "target_transform": "train-only standardization, report in MPa",
        "counts": COUNTS, "kernels": KERNELS, "mlp_widths": [8, 27, 26, 1],
        "development_initializations": DEV_INITS, "final_initializations": FINAL_INITS, "rates": RATES,
        "training": {"optimizer": "Adam", "dtype": "float64", "batch": BATCH, "epochs": EPOCHS,
                     "patience": PATIENCE, "regularization": 0, "batch_stream": "SeedSequence([31431, fold, init])",
                     "selection": "earliest validation RMSE minimum, mean over development initializations"},
        "runtime": {**_runtime(), "device": "cpu", "threads": 1, "workers": scalar.WORKERS},
        "code_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(scalar.__file__), Path(base.__file__))}
    }
    protocol = json.loads(json.dumps(protocol))
    protocol["id"] = _digest(protocol)
    output = Path(root) / protocol["id"][:16]
    path = output / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        _write_json(path, protocol)
    return output, protocol


def _split(rows):
    x, y, groups = _dataset()
    train = rows["train"]
    assert sorted(i for values in rows.values() for i in values) == list(range(len(x)))
    assert sum(len(np.unique(groups[i])) for i in rows.values()) == len(np.unique(groups))
    center, scale = x[train].mean(0), x[train].std(0)
    y_center, y_scale = y[train].mean(), y[train].std()
    assert (scale > 0).all() and y_scale > 0
    result = {k: (torch.from_numpy((x[v] - center) / scale), torch.from_numpy((y[v] - y_center) / y_scale))
              for k, v in rows.items()}
    scaling = {"x_mean": center.tolist(), "x_sd": scale.tolist(), "y_mean": float(y_center), "y_sd": float(y_scale)}
    return result, scaling


def _batch_seed(fold, init):
    return int(np.random.SeedSequence([31431, fold, init]).generate_state(1)[0])


def _train(model, identity, data, limit):
    x, y = data["train"]
    vx, vy = data["val"]
    optimizer = torch.optim.Adam(model.parameters(), lr=identity["lr"])
    rng = torch.Generator().manual_seed(_batch_seed(identity["fold"], identity["init"]))
    history, best = [], math.inf
    started = perf_counter()
    for epoch in range(1, limit + 1):
        for ii in torch.randperm(len(x), generator=rng).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            (model(x[ii]) - y[ii]).square().mean().backward()
            optimizer.step()
        with torch.no_grad():
            value = rmse(model(vx), vy)
        assert math.isfinite(value)
        history.append(value)
        if value < best:
            best, selected = value, epoch
            checkpoint = deepcopy(model.state_dict())
        if epoch - selected == PATIENCE:
            break
    return {"checkpoint": checkpoint, "best_epoch": selected, "validation_history": history,
            "seconds": perf_counter() - started}


@torch.inference_mode()
def _attach_test(fit, inputs, indices, scaling):
    model = make_model(fit["identity"]["model"], fit["identity"]["init"])
    model.load_state_dict(fit["checkpoint"], strict=True)
    pred = model(inputs) * scaling["y_sd"] + scaling["y_mean"]
    target = torch.from_numpy(_dataset()[1][indices])
    fit.update(test={"rows": indices, "prediction": pred, "target": target},
               rmse=rmse(pred, target), mae=(pred-target).abs().mean().item())


@torch.inference_mode()
def _verify(fit, identity, data, scaling, limit):
    assert fit["identity"] == identity and fit["scaling"] == scaling
    assert math.isfinite(fit["seconds"]) and fit["seconds"] >= 0
    model = make_model(identity["model"], identity["init"])
    model.load_state_dict(fit["checkpoint"], strict=True)
    assert parameter_count(model) == COUNTS[identity["model"]]
    assert all(torch.isfinite(p).all() for p in model.parameters())
    history = fit["validation_history"]
    best = min(range(len(history)), key=history.__getitem__) + 1
    assert math.isfinite(sum(history)) and fit["best_epoch"] == best
    assert len(history) == min(limit, best + PATIENCE)
    x, y = data["val"]
    assert rmse(model(x), y) == history[best-1]
    if identity["stage"] == "final":
        test = fit["test"]
        assert test["rows"] == identity["test_rows"]
        assert torch.equal(test["target"], torch.from_numpy(_dataset()[1][test["rows"]]))
        assert torch.equal(test["prediction"], model(data["test"][0]) * scaling["y_sd"] + scaling["y_mean"])
        assert rmse(test["prediction"], test["target"]) == fit["rmse"]
        assert (test["prediction"] - test["target"]).abs().mean().item() == fit["mae"]
        assert math.isfinite(fit["rmse"] + fit["mae"])
    else:
        assert "test" not in fit


def _job(args):
    output, protocol, stage, fold, model, init, lr = args
    key = f"fold{fold+1}-{model}-{init}-lr{lr:g}"
    path = output / stage / "fits" / f"{key}.pt"
    rows = protocol["folds"][fold]
    data, scaling = _split(rows)
    identity = {"protocol_id": protocol["id"], "stage": stage, "fold": fold,
                "model": model, "init": init, "lr": lr}
    if stage == "final":
        identity["test_rows"] = rows["test"]
    cached = path.exists()
    if cached:
        fit = torch.load(path, weights_only=True, map_location="cpu")
    else:
        fit = _train(make_model(model, init), identity, data, EPOCHS)
        fit.update(identity=identity, scaling=scaling)
        if stage == "final":
            _attach_test(fit, data["test"][0], rows["test"], scaling)
        base._save_pt(path, fit)
        fit = torch.load(path, weights_only=True, map_location="cpu")
    _verify(fit, identity, data, scaling, EPOCHS)
    row = {"key": key, "fold": fold, "model": model, "init": init, "lr": lr,
           "parameters": COUNTS[model], "seconds": fit["seconds"], "best_epoch": fit["best_epoch"],
           "val_rmse": fit["validation_history"][fit["best_epoch"] - 1] * scaling["y_sd"],
           "artifact": str(path.relative_to(output)), "sha256": sha256(path.read_bytes()).hexdigest()}
    if stage == "final":
        row.update(rmse=fit["rmse"], mae=fit["mae"])
    return row, cached


def _run(output, protocol, stage, jobs):
    assert json.loads((output / "protocol.json").read_text()) == protocol
    start, rows, reused = perf_counter(), [], 0
    with ProcessPoolExecutor(max_workers=scalar.WORKERS, mp_context=get_context("spawn"),
                             initializer=_setup_torch) as pool:
        for row, cached in pool.map(_job, [(output, protocol, stage, *j) for j in jobs]):
            rows.append(row)
            reused += cached
            if len(rows) % 15 == 0 or len(rows) == len(jobs):
                print(f"{stage}: {len(rows)}/{len(jobs)} verified ({reused} reused)", flush=True)
    assert len({r["key"] for r in rows}) == len(jobs)
    assert {p.stem for p in (output / stage / "fits").glob("*.pt")} == {r["key"] for r in rows}
    manifest = {"protocol_id": protocol["id"], "fits": rows}
    path = output / stage / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        _write_json(path, manifest)
    print(f"{stage}: {perf_counter()-start:.1f} s on CPU", flush=True)
    return rows


def checks(output, protocol):
    _setup_torch()
    assert {m: parameter_count(make_model(m, FINAL_INITS[0])) for m in COUNTS} == COUNTS
    for rows in protocol["folds"]:
        _split(rows)
    data, scaling = _split(protocol["folds"][0])
    smoke = {}
    with TemporaryDirectory() as folder:
        for model in COUNTS:
            identity = {"stage": "smoke", "model": model, "init": FINAL_INITS[0], "lr": .003, "fold": 0}
            fit = _train(make_model(model, FINAL_INITS[0]), identity, data, 2)
            fit.update(identity=identity, scaling=scaling)
            path = Path(folder) / "smoke.pt"
            base._save_pt(path, fit)
            _verify(torch.load(path, weights_only=True, map_location="cpu"), identity, data, scaling, 2)
            smoke[model] = fit["validation_history"][fit["best_epoch"] - 1] * scaling["y_sd"]
    result = {"rows": protocol["rows"], "groups": protocol["group_count"], "parameters": COUNTS,
              "group_leaks": 0, "smoke_validation_rmse_mpa": smoke}
    _write_json(output / "checks.json", result)
    return result


def tune(output, protocol):
    rows = _run(output, protocol, "development",
                list(product(range(FOLDS), COUNTS, DEV_INITS, RATES)))
    assert len(rows) == 90
    chosen = []
    for fold in range(FOLDS):
        rates = {}
        for model in COUNTS:
            results = []
            for lr in RATES:
                values = [r["val_rmse"] for r in rows
                          if (r["fold"], r["model"], r["lr"]) == (fold, model, lr)]
                assert len(values) == len(DEV_INITS)
                results.append((mean(values), lr))
            rates[model] = min(results)[1]
        chosen.append(rates)
    choice = {"protocol_id": protocol["id"], "selected": chosen}
    path = output / "development" / "selection.json"
    if path.exists():
        assert json.loads(path.read_text()) == choice
    else:
        _write_json(path, choice)
    return choice


def run_final(output, protocol, choice):
    assert choice == json.loads((output / "development" / "selection.json").read_text())
    jobs = [(fold, model, init, choice["selected"][fold][model])
            for fold, model, init in product(range(FOLDS), COUNTS, FINAL_INITS)]
    rows = _run(output, protocol, "final", jobs)
    assert len(rows) == 45
    return rows


def report(rows, protocol):
    from IPython.display import Markdown, display

    values = {}
    for fold, model in product(range(FOLDS), COUNTS):
        fits = [r for r in rows if (r["fold"], r["model"]) == (fold, model)]
        assert len(fits) == len(FINAL_INITS)
        values[fold, model] = [mean(r[k] for r in fits) for k in ("rmse", "mae")]
    display(Markdown("### Прогноз прочности на отложенных составах"))
    base._table(("Модель", "Параметры", "Средняя RMSE, МПа", "Средняя MAE, МПа"),
                [(m, COUNTS[m], *(f"{mean(values[f, m][i] for f in range(FOLDS)):.3f}" for i in range(2)))
                 for m in COUNTS])
    display(Markdown("### Ошибки по внешним блокам и парные разности"))
    base._table(("Блок", "Тестовых строк", "II RMSE / MAE", "III RMSE / MAE", "MLP RMSE / MAE",
                 "III−II RMSE", "III−MLP RMSE"),
                [(f+1, len(protocol["folds"][f]["test"]),
                  *(" / ".join(f"{x:.3f}" for x in values[f, m]) for m in COUNTS),
                  *(f"{values[f,'III'][0]-values[f,m][0]:+.3f}" for m in ("II", "MLP")))
                 for f in range(FOLDS)])
