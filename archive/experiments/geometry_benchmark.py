"""Seventh-arm extension of the original Gaussian geometry benchmark."""

import ast
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import inspect
from itertools import combinations, product
import json
import math
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from torch import nn

from experiments import geometry_reference as ref
from src.experiment import _atomic_file, _digest, _runtime, _write_json
from src.models import MGS_KAN

MODEL = "MGS-KAN"
PARAMETERS = 2515
SINGLE_KERNELS, PAIR_KERNELS, PENALTY = 3, 5, 0.001
REFERENCE = Path("gaussian_kan_geometry_vs_mlp_v2.ipynb")
ARCHIVE = Path("gkan_results/main")
OUTPUT = Path("outputs/study/geometry_benchmark")
REFERENCE_CELLS = (3, 5, 9, 15, 17, 19, 21)
RTOL, ATOL = 1e-9, 1e-12


class MGSResidual(nn.Module):
    """Shared centers and precisions with five independent output amplitudes."""

    factor_entries = MGS_KAN.factor_entries

    def __init__(self):
        super().__init__()
        self.register_buffer("pairs", torch.triu_indices(10, 10, offset=1).T)
        self.affine = nn.Linear(10, 5)
        nn.init.normal_(self.affine.weight, std=0.01)
        nn.init.zeros_(self.affine.bias)
        self.single_center = nn.Parameter(torch.linspace(-0.75, 0.75, SINGLE_KERNELS).repeat(10, 1)
                                         + 0.025 * torch.randn(10, SINGLE_KERNELS))
        self.single_log_scale = nn.Parameter(torch.zeros(10, SINGLE_KERNELS))
        self.single_amplitude = nn.Parameter(0.003 * torch.randn(10, SINGLE_KERNELS, 5))
        anchors = torch.tensor([[-0.6, -0.6], [-0.6, 0.6], [0.0, 0.0], [0.6, -0.6], [0.6, 0.6]])
        self.pair_center = nn.Parameter(anchors + 0.03 * torch.randn(45, PAIR_KERNELS, 2))
        self.pair_precision_raw = nn.Parameter(0.08 * torch.randn(45, PAIR_KERNELS, 3))
        self.pair_amplitude = nn.Parameter(0.003 * torch.randn(45, PAIR_KERNELS, 5))

    def forward(self, x):
        delta = (x.unsqueeze(-1) - self.single_center) * self.single_log_scale.exp()
        singles = torch.exp(-0.5 * delta.square())
        i, j = self.pairs.T
        u, v = (x[:, i].unsqueeze(-1), x[:, j].unsqueeze(-1))
        cu, cv = self.pair_center.unbind(-1)
        du, dv = u - cu, v - cv
        a, b, d = self.factor_entries()
        # Full precision P = L Lᵀ, evaluated as ||Lᵀ (x - center)||².
        pairs = torch.exp(-0.5 * ((a * du + b * dv).square() + (d * dv).square()))
        return (self.affine(x) + singles.flatten(1) @ self.single_amplitude.flatten(0, 1)
                + pairs.flatten(1) @ self.pair_amplitude.flatten(0, 1))

    def group_penalty(self):
        return torch.linalg.vector_norm(self.pair_amplitude.flatten(1), dim=1).sum()


class GeometryMGSKAN(nn.Module):
    """FlatMLP's Gaussian interface with an explicit Gaussian residual."""

    def __init__(self):
        super().__init__()
        self.register_buffer("coord_scale", torch.tensor([1.5, 1.5, .7, .7, .7]))
        self.mgs_residual = MGSResidual()

    def forward(self, mu, cov):
        x = torch.tanh(ref.encode_gaussian(mu, cov) / self.coord_scale).flatten(1)
        base = ref.encode_gaussian(mu.mean(1), cov.mean(1))
        pred_mu, pred_cov = ref.decode_gaussian(base + self.mgs_residual(x))
        return pred_mu[:, None], pred_cov[:, None]


@dataclass(frozen=True)
class Study:
    output: Path
    notebook: Path
    archive: Path
    protocol: dict

    @property
    def id(self):
        return _digest(self.protocol)

    @property
    def device(self):
        return torch.device(self.protocol["runtime"]["device"])


def _source_cells(notebook):
    cells = json.loads(notebook.read_text())["cells"]
    return {i: "".join(cells[i]["source"]) for i in REFERENCE_CELLS}


def _run_key(cfg):
    return f"{cfg.task}__d{cfg.data_seed}__s{cfg.train_seed}__lr{cfg.lr:g}__{cfg.config_id}"


def _historic_rows(archive):
    rows = [json.loads(line) for line in (archive / "runs.jsonl").read_text().splitlines() if line.strip()]
    return [row for row in rows if row["stage"] == "robustness" and row["status"] == "done"]


def _archive_selection(archive):
    return json.loads((archive / "selected_lr.json").read_text())["selection"]


def _historical_checkpoint(archive, row):
    return archive / "ckpt" / row["run_id"] / "best.pt"


def _archive_inputs(archive, rows):
    return {
        "records_sha256": _digest(sorted(rows, key=lambda row: row["run_id"])),
        "settings_sha256": _digest(_archive_selection(archive)),
        "checkpoints": {row["run_id"]: sha256(_historical_checkpoint(archive, row).read_bytes()).hexdigest()
                        for row in rows},
    }


def freeze_protocol(root=OUTPUT, notebook=REFERENCE, archive=ARCHIVE):
    notebook, archive = Path(notebook), Path(archive)
    runtime = _runtime()
    cuda = torch.cuda.is_available()
    runtime.update(device="cuda" if cuda else "cpu", dtype="float64", cuda=torch.version.cuda,
                   device_name=torch.cuda.get_device_name() if cuda else runtime["cpu"])
    protocol = {
        "model": MODEL, "parameters": PARAMETERS, "single_kernels": SINGLE_KERNELS,
        "pair_kernels": PAIR_KERNELS, "group_penalty": PENALTY, "split_sizes": ref.SIZES0,
        "stage_b": {k: v for k, v in ref.STAGES["lr_search"].items() if k not in ("device", "models")},
        "stage_d": {k: v for k, v in ref.STAGES["robustness"].items() if k not in ("device", "models")},
        "adam_betas": [0.9, 0.999], "adam_eps": 1e-8, "clip_norm": 5.0,
        "checkpoint": "earliest minimum unpenalized validation W2 squared (including step 0)",
        "learning_rate": "minimum mean best validation W2 squared, ties go to the smaller rate",
        "batch_rng": "CPU torch.Generator seeded by training seed + 10000, draws with replacement",
        "historical": _archive_inputs(archive, _historic_rows(archive)),
        "runtime": runtime, "threads_per_worker": 1, "workers": 1 if cuda else 5,
        "code_sha256": {Path(__file__).name: sha256(Path(__file__).read_bytes()).hexdigest(),
                        Path(ref.__file__).name: sha256(Path(ref.__file__).read_bytes()).hexdigest(),
                        "notebook_reference_cells": _digest(_source_cells(notebook)),
                        "shared_helpers": _digest([inspect.getsource(f) for f in
                                                   (_atomic_file, _digest, _runtime, _write_json, MGS_KAN.factor_entries)])},
    }
    protocol = json.loads(json.dumps(protocol))
    output = Path(root) / _digest(protocol)[:16]
    path = output / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        _write_json(path, protocol)
    return Study(output, notebook, archive, protocol)


def _configure_worker():
    torch.set_num_threads(1)


def _new_model(seed, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return GeometryMGSKAN().to(device)


def _on_device(split, device):
    return tuple(tuple(t.to(device) for t in tensors) for tensors in split)


@torch.no_grad()
def _metrics(pred, target):
    fields = ref.batched_metrics(tuple(x[None] for x in pred), tuple(x[None] for x in target))[0].tolist()
    values = dict(zip(ref.METRIC_NAMES, fields))
    assert all(math.isfinite(v) for v in fields)
    assert values["finite"] == 1 and values["min_eig"] > 0
    assert math.isclose(values["w2"], values["mean_error"] + values["cov_error"], abs_tol=ATOL, rel_tol=RTOL)
    return values


@torch.no_grad()
def _curve_row(model, train, validation):
    G = tuple(torch.cat((x, y)) for x, y in zip(train[0], validation[0]))
    Y = tuple(torch.cat((x, y)) for x, y in zip(train[1], validation[1]))
    em, ec = ref.w2_parts(model(*G), Y)
    n = len(train[0][0])
    numbers = []
    for cut in (slice(0, n), slice(n, None)):
        a, b = em[cut], ec[cut]
        numbers.extend(((a + b).mean(), a.mean(), b.mean()))
    row = dict(zip(ref.CURVE_COLS, torch.stack(numbers).tolist()))
    assert all(math.isfinite(v) for v in row.values())
    return row


def _save_pt(path, data):
    with _atomic_file(path) as temp:
        torch.save(data, temp)


def _identity(study, cfg):
    return {"protocol_id": study.id, "key": _run_key(cfg), "config": json.loads(json.dumps(asdict(cfg)))}


def _fit(study, cfg, dataset, partial, stop_at=None):
    device = study.device
    model = _new_model(cfg.train_seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.adam_betas, eps=cfg.adam_eps)
    rng = torch.Generator(device="cpu").manual_seed(cfg.train_seed + 10000)
    train, validation = (_on_device(dataset[split], device) for split in ("train", "val"))
    G, Y = train
    identity = _identity(study, cfg)
    initial, seconds, history, best_val, best = 0, 0.0, [], math.inf, None
    if partial.exists():
        saved = torch.load(partial, weights_only=True, map_location="cpu")
        if saved["identity"] != identity:
            raise ValueError(f"Partial checkpoint protocol mismatch: {partial}")
        initial, seconds, history = saved["step"], saved["seconds"], saved["history"]
        assert 0 < initial < cfg.steps and initial % cfg.eval_every == 0 and history[-1]["step"] == initial
        rng.set_state(saved["rng"])
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        best_step, best_val, best = saved["best_step"], saved["best_val"], saved["best"]
        if device.type != "cpu":
            best = {key: value.to(device) for key, value in best.items()}
    ref.sync(device)
    started = perf_counter()
    checkpoint_every = study.protocol["stage_d"]["save_every"]
    for step in range(initial, cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            if not (initial and step == initial):
                entry = {"step": step, **_curve_row(model, train, validation)}
                history.append(entry)
                if entry["val_w2"] < best_val:
                    best_val, best_step, best = entry["val_w2"], step, deepcopy(model.state_dict())
        if step == cfg.steps:
            break
        if step > initial and (step % checkpoint_every == 0 or step == stop_at):
            ref.sync(device)
            seconds += perf_counter() - started
            _save_pt(partial, {"identity": identity, "step": step, "seconds": seconds, "history": history,
                               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "rng": rng.get_state(), "best_val": best_val, "best_step": best_step, "best": best})
            if step == stop_at:
                return None
            started = perf_counter()
        idx = ref.batch_indices(cfg.n_train, cfg.batch, 1, rng)[0].to(device)
        pred = model(*(tensor[idx] for tensor in G))
        loss = ref.w2sq(pred, tuple(tensor[idx] for tensor in Y)).mean()
        loss = loss + study.protocol["group_penalty"] * model.mgs_residual.group_penalty()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm, error_if_nonfinite=True)
        optimizer.step()
    ref.sync(device)
    seconds += perf_counter() - started
    model.load_state_dict(best, strict=True)
    return model, best_step, history, seconds


@torch.inference_mode()
def _save_result(path, study, cfg, dataset, fit, include_test):
    model, best_step, history, seconds = fit
    selected, metrics = {}, {}
    splits = ref.task_splits(cfg.task) if include_test else ("train", "val")
    for split in splits:
        G, Y = _on_device(dataset[split], study.device)
        prediction = model(*G)
        metrics[split] = _metrics(prediction, Y)
        selected[split] = {"prediction": tuple(x.cpu() for x in prediction),
                           "target": tuple(x.cpu() for x in Y)}
    bundle = {"identity": _identity(study, cfg), "stage": "D" if include_test else "B",
              "checkpoint": {k: v.cpu() for k, v in model.state_dict().items()}, "best_step": best_step,
              "history": history, "training_seconds": seconds, "metrics": metrics, "splits": selected}
    _save_pt(path, bundle)


@torch.inference_mode()
def _verify_result(path, study, cfg, dataset, include_test):
    bundle = torch.load(path, weights_only=True, map_location="cpu")
    assert bundle["identity"] == _identity(study, cfg)
    assert bundle["stage"] == ("D" if include_test else "B")
    expected = list(range(0, cfg.steps + 1, cfg.eval_every))
    if expected[-1] != cfg.steps:
        expected.append(cfg.steps)
    history = bundle["history"]
    assert [r["step"] for r in history] == expected
    assert all(math.isfinite(r[key]) for r in history for key in ref.CURVE_COLS)
    best = min(history, key=lambda r: (r["val_w2"], r["step"]))
    assert bundle["best_step"] == best["step"]
    model = _new_model(cfg.train_seed, torch.device("cpu"))
    model.load_state_dict(bundle["checkpoint"], strict=True)
    assert ref.count_params(model) == PARAMETERS and all(torch.isfinite(p).all() for p in model.parameters())
    splits = ref.task_splits(cfg.task) if include_test else ["train", "val"]
    assert set(bundle["metrics"]) == set(bundle["splits"]) == set(splits)
    for split in splits:
        G, target = dataset[split]
        saved = bundle["splits"][split]
        for expected_y, observed_y in zip(target, saved["target"], strict=True):
            assert torch.equal(expected_y, observed_y)
        for actual, previous in zip(model(*G), saved["prediction"], strict=True):
            torch.testing.assert_close(actual, previous, rtol=RTOL, atol=ATOL)
        restored = _metrics(saved["prediction"], saved["target"])
        for name, value in restored.items():
            assert math.isclose(value, bundle["metrics"][split][name], rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(best["val_w2"], bundle["metrics"]["val"]["w2"], rel_tol=RTOL, abs_tol=ATOL)
    assert math.isfinite(bundle["training_seconds"]) and bundle["training_seconds"] >= 0
    return {"key": _run_key(cfg), "task": cfg.task, "model": MODEL, "data_seed": cfg.data_seed,
            "train_seed": cfg.train_seed, "lr": cfg.lr, "parameters": PARAMETERS,
            "selected_step": best["step"], "metrics": bundle["metrics"],
            "training_seconds": bundle["training_seconds"], "device": study.protocol["runtime"]["device_name"],
            "provenance": "new fit", "artifact": str(path.resolve().relative_to(study.output.resolve())),
            "sha256": sha256(path.read_bytes()).hexdigest()}


def _job(args):
    study, cfg, include_test = args
    stage = "stage_d" if include_test else "stage_b"
    path = study.output / stage / f"{_run_key(cfg)}.pt"
    partial = path.with_suffix(".partial.pt")
    with ref.float64():
        dataset = ref.make_dataset(cfg.task, cfg.data_seed, cfg.n_train, cfg.n_val, cfg.n_test)
        reused = False
        if path.exists():
            try:
                row = _verify_result(path, study, cfg, dataset, include_test)
                reused = True
            except (AssertionError, RuntimeError, ValueError, KeyError, EOFError) as error:
                print(f"Rebuilding unverified fit {path.name}: {error}", flush=True)
        if not reused:
            fit = _fit(study, cfg, dataset, partial)
            _save_result(path, study, cfg, dataset, fit, include_test)
            row = _verify_result(path, study, cfg, dataset, include_test)
        if partial.exists():
            partial.unlink()
    return row, reused


def _configs(study, phase, selected=None):
    stage = study.protocol[phase]
    settings = {k: stage[k] for k in ref.RUN_FIELDS}
    for task, data_seed, train_seed in product(stage["tasks"], stage["data_seeds"], stage["train_seeds"]):
        rates = stage["lrs"] if selected is None else [selected[task]]
        for lr in rates:
            yield ref.RunCfg(MODEL, task, data_seed, train_seed, lr, **settings)


def _run(study, configurations, include_test):
    configs = list(configurations)
    target = 27 if include_test else 18
    assert len(configs) == len({_run_key(c) for c in configs}) == target
    jobs = [(study, cfg, include_test) for cfg in configs]
    rows, reused = [], 0
    with ProcessPoolExecutor(max_workers=study.protocol["workers"], mp_context=get_context("spawn"),
                             initializer=_configure_worker) as pool:
        for i, (row, cached) in enumerate(pool.map(_job, jobs), 1):
            rows.append(row)
            reused += cached
            if i % 3 == 0:
                print(f"{i}/{target} {'Stage D' if include_test else 'Stage B'} fits verified ({reused} reused)", flush=True)
    stage = "stage_d" if include_test else "stage_b"
    assert {p.stem for p in (study.output / stage).glob("*.pt")} == {r["key"] for r in rows}
    _write_json(study.output / stage / "manifest.json", {"protocol_id": study.id, "fits": rows})
    return rows


def tune(study):
    rows = _run(study, _configs(study, "stage_b"), include_test=False)
    candidates, selected = [], {}
    for task in ref.TASKS:
        scores = []
        for lr in study.protocol["stage_b"]["lrs"]:
            subset = [r for r in rows if r["task"] == task and r["lr"] == lr]
            assert {r["train_seed"] for r in subset} == {11, 29} and len(subset) == 2
            score = mean(r["metrics"]["val"]["w2"] for r in subset)
            scores.append((score, lr))
            candidates.append({"task": task, "lr": lr, "mean_best_val_w2": score})
        selected[task] = min(scores)[1]
    choice = {"protocol_id": study.id, "selected": selected, "candidates": candidates}
    path = study.output / "selected_lr.json"
    if path.exists():
        assert json.loads(path.read_text()) == choice
    else:
        _write_json(path, choice)
    return rows, choice


def run_final(study, choice):
    assert choice == json.loads((study.output / "selected_lr.json").read_text())
    assert choice["protocol_id"] == study.id
    return _run(study, _configs(study, "stage_d", choice["selected"]), include_test=True)


def _notebook_scope(notebook):
    """Evaluate selected executable definitions for parity, without running cells."""
    scope = {"torch": torch, "nn": nn, "math": math, "hashlib": __import__("hashlib"),
             "json": json, "dataclass": dataclass, "asdict": asdict, "DEVICE": torch.device("cpu"),
             "DTYPE": torch.float64, "INTERVALS": 8, "DEGREE": 3, "WIDTHS": (2, 3, 1)}
    notebook_sources = _source_cells(notebook)
    for i, source in notebook_sources.items():
        nodes = []
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and hasattr(ref, node.name):
                nodes.append(node)
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and hasattr(ref, node.targets[0].id)):
                nodes.append(node)
        if nodes:
            exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{notebook}:cell{i}", "exec"), scope)
    return scope, notebook_sources[19]


def _reference_checks(study):
    original, training_source = _notebook_scope(study.notebook)
    for task, seed in product(ref.TASKS, (2000, 3000)):
        actual = ref.make_dataset(task, seed, *ref.SIZES0)
        expected = original["make_dataset"](task, seed, *ref.SIZES0)
        assert list(actual) == list(expected)
        for split in expected:
            for tensors_a, tensors_b in zip(actual[split], expected[split], strict=True):
                for a, b in zip(tensors_a, tensors_b, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
    group = next(n for n in ast.parse(training_source).body if isinstance(n, ast.FunctionDef) and n.name == "train_group")
    chunk = next(n for n in ast.walk(group) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "chunk")
    state = {"torch": torch, "n_train": 256, "size": 17, "c0": ref.RunCfg("Flat MLP", "affine", 2000,
             29, 0.003, 2000, 64, *ref.SIZES0, 12), "device": torch.device("cpu"),
             "gens": [torch.Generator().manual_seed(10029)]}
    exec(compile(ast.Module(body=[chunk], type_ignores=[]), f"{study.notebook}:batch_indices", "exec"), state)
    actual_batch = ref.batch_indices(256, 64, 17, torch.Generator().manual_seed(10029))
    assert torch.equal(actual_batch, state["chunk"][0])
    pair = ref.make_dataset("geodesic", 5000, *ref.SIZES0)["test"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        executable = original["FlatMLP"]()
    extracted = ref.FlatMLP()
    extracted.load_state_dict(executable.state_dict())
    for actual, expected in zip(extracted(*pair[0]), executable(*pair[0]), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual, expected = ref.evaluate(extracted, pair), original["evaluate"](executable, pair)
    assert actual == expected
    assert ref.STAGES["lr_search"]["train_seeds"] == [11, 29]
    assert ref.STAGES["robustness"]["train_seeds"] == [11, 29, 47]
    return actual["w2"]


@torch.inference_mode()
def _verify_archive(study):
    archived = _historic_rows(study.archive)
    settings = {(r["task"], r["model"]): r["lr"] for r in _archive_selection(study.archive)}
    index = {(r["task"], r["model"], r["data_seed"], r["train_seed"]): r for r in archived}
    stage = study.protocol["stage_d"]
    expected = set(product(stage["tasks"], ref.MODELS, stage["data_seeds"], stage["train_seeds"]))
    if len(archived) != len(index) or set(index) != expected:
        raise ValueError(f"Historical Stage D archive is incomplete or ambiguous. Missing: {sorted(expected - set(index))}")
    assert study.protocol["historical"] == _archive_inputs(study.archive, archived)
    replay_error, rows = 0.0, []
    for key in sorted(index):
        task, model_name, data_seed, train_seed = key
        row = index[key]
        cfg = ref.RunCfg(**row["config"])
        assert row["run_id"] == cfg.run_id and (cfg.task, cfg.model, cfg.data_seed, cfg.train_seed) == key
        assert cfg.lr == row["lr"] == settings[task, model_name]
        assert all(getattr(cfg, k) == stage[k] for k in ref.RUN_FIELDS)
        assert cfg.clip_norm == study.protocol["clip_norm"] and list(cfg.adam_betas) == study.protocol["adam_betas"]
        assert cfg.adam_eps == study.protocol["adam_eps"] and cfg.protocol == "gkan-v2"
        checkpoint = _historical_checkpoint(study.archive, row)
        bundle = torch.load(checkpoint, weights_only=True, map_location="cpu")
        assert bundle["run_id"] == row["run_id"] and json.loads(bundle["config"]) == row["config"]
        assert bundle["step"] == row["best_step"] and bundle["val_w2"] == row["best_val_w2"]
        model = ref.build_model(model_name, cfg.bary_iters)
        model.load_state_dict(bundle["state_dict"], strict=True)
        assert ref.count_params(model) == row["params"]
        dataset = ref.make_dataset(task, data_seed, *ref.SIZES0)
        metrics = row["metrics"]["best"]
        assert set(metrics) == set(ref.task_splits(task))
        for split in ref.task_splits(task):
            G, Y = dataset[split]
            recomputed = _metrics(model(*G), Y)
            saved = metrics[split]
            for name in ref.METRIC_NAMES:
                assert math.isclose(saved[name], recomputed[name], rel_tol=RTOL, abs_tol=ATOL), (row["run_id"], split, name)
            assert math.isclose(saved["w2"], saved["mean_error"] + saved["cov_error"], rel_tol=RTOL, abs_tol=ATOL)
            replay_error = max(replay_error, abs(saved["w2"] - recomputed["w2"]))
        rows.append({"key": row["run_id"], "task": task, "model": model_name, "data_seed": data_seed,
                     "train_seed": train_seed, "lr": row["lr"], "parameters": row["params"],
                     "selected_step": row["best_step"], "metrics": metrics,
                     "training_seconds": row["seconds_amortized"], "device": row["device_name"],
                     "provenance": "archived Stage D, CPU checkpoint replay verified",
                     "artifact": str(checkpoint.relative_to(study.archive)),
                     "sha256": study.protocol["historical"]["checkpoints"][row["run_id"]]})
    result = {"protocol_id": study.id, "archive": str(study.archive), "fits": rows,
              "cpu_replay_max_abs_w2_difference": replay_error}
    _write_json(study.output / "historical.json", result)
    return result


def run_checks(study):
    from tempfile import TemporaryDirectory

    _configure_worker()
    with ref.float64():
        benchmark_w2 = _reference_checks(study)
        device = study.device
        model = _new_model(11, device)
        twin = _new_model(11, device)
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor, twin.state_dict()[key])
        assert ref.count_params(model) == PARAMETERS
        assert model.mgs_residual.pairs.cpu().tolist() == [list(pair) for pair in combinations(range(10), 2)]
        G, Y = _on_device(ref.make_dataset("geodesic", 2000, *ref.SIZES0)["val"], device)
        pred = model(*G)
        assert pred[0].shape == (128, 1, 2) and pred[1].shape == (128, 1, 2, 2)
        assert model.coord_scale.dtype == torch.float64
        assert bool((ref.eigmin_sym2(pred[1]) > 0).all())
        ref.w2sq(pred, Y).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        with torch.no_grad():
            twin.mgs_residual.affine.weight.zero_()
            twin.mgs_residual.affine.bias.zero_()
            twin.mgs_residual.single_amplitude.zero_()
            twin.mgs_residual.pair_amplitude.zero_()
            base = ref.decode_gaussian(ref.encode_gaussian(G[0].mean(1), G[1].mean(1)))
            for actual, expected in zip(twin(*G), base, strict=True):
                torch.testing.assert_close(actual, expected[:, None], rtol=0, atol=0)
        cfg = replace(next(_configs(study, "stage_b")), steps=20)
        with TemporaryDirectory(dir=study.output) as folder:
            folder = Path(folder)
            dataset = ref.make_dataset(cfg.task, cfg.data_seed, *ref.SIZES0)
            partial = folder / "smoke.partial.pt"
            assert _fit(study, cfg, dataset, partial, stop_at=10) is None
            resumed = _fit(study, cfg, dataset, partial)
            direct = _fit(study, cfg, dataset, folder / "direct.partial.pt")
            assert resumed[1:3] == direct[1:3]
            for key, tensor in resumed[0].state_dict().items():
                assert torch.equal(tensor, direct[0].state_dict()[key])
            save_path = folder / "smoke.pt"
            _save_result(save_path, study, cfg, dataset, resumed, include_test=True)
            _verify_result(save_path, study, cfg, dataset, include_test=True)
        historical = _verify_archive(study)
    result = {"passed": True, "reference_dataset_task_seed_checks": 6, "batch_indices_checked": 17 * 64,
              "reference_flat_geodesic_fixture_w2": benchmark_w2, "trainable_parameters": PARAMETERS,
              "smoke_updates": 20, "partial_resume_exact": True, "archived_replays": len(historical["fits"]),
              "archive_cpu_replay_max_abs_w2_difference": historical["cpu_replay_max_abs_w2_difference"]}
    _write_json(study.output / "checks.json", result)
    return result


def _table(headers, rows):
    from IPython.display import Markdown, display

    lines = [[*headers], ["---"] * len(headers), *rows]
    display(Markdown("\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in lines)))


def _fit_csv(output, rows):
    fields = ("key", "task", "model", "data_seed", "train_seed", "lr", "parameters", "selected_step",
              "device", "provenance", "artifact", "sha256", "training_seconds", "split", *ref.METRIC_NAMES)
    with _atomic_file(output / "individual_fits.csv") as temp:
        with temp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                common = {k: row[k] for k in fields if k in row}
                for split, metrics in row["metrics"].items():
                    writer.writerow({**common, "split": split, **metrics})


def _aggregates(rows, task, model, split="test", seed=None):
    subset = [r for r in rows if r["task"] == task and r["model"] == model
              and (seed is None or r["data_seed"] == seed)]
    assert len(subset) == (9 if seed is None else 3)
    return tuple(mean(r["metrics"][split][k] for r in subset) for k in ("w2", "mean_error", "cov_error"))


def show_tuning(choice):
    _table(("Task", "Learning rate", "Mean best validation W₂²", "Selected"), [
        (r["task"], r["lr"], f"{r['mean_best_val_w2']:.7g}",
         "Yes" if r["lr"] == choice["selected"][r["task"]] else "")
        for r in choice["candidates"]
    ])


def report(study, final):
    from IPython.display import Markdown, display

    legacy = json.loads((study.output / "historical.json").read_text())
    assert legacy["protocol_id"] == study.id and len(legacy["fits"]) == 162 and len(final) == 27
    all_rows = final + legacy["fits"]
    _fit_csv(study.output, all_rows)
    models = [MODEL, *ref.MODELS]
    main, paired, shifted, runtime, verdict = [], [], [], [], []
    seeds = study.protocol["stage_d"]["data_seeds"]
    for task in ref.TASKS:
        for model in models:
            w2, mc, cc = _aggregates(all_rows, task, model)
            count = next(r["parameters"] for r in all_rows if r["model"] == model)
            main.append((task, model, count, f"{w2:.7f}", f"{mc:.7f}", f"{cc:.7f}"))
            ood = [_aggregates(all_rows, task, model, split)[0] for split in ref.OOD_VARIANTS]
            shifted.append((task, model, *(f"{v:.7f}" for v in ood)))
        comparators = ["Flat MLP", "Spline / mean", "Spline / BW"]
        if task == "geodesic":
            comparators.append("MLP edges / BW")
        for rival in comparators:
            delta = [_aggregates(all_rows, task, MODEL, seed=s)[0] -
                     _aggregates(all_rows, task, rival, seed=s)[0] for s in seeds]
            new, old = (_aggregates(all_rows, task, arm) for arm in (MODEL, rival))
            avg, center, covariance = (a - b for a, b in zip(new, old, strict=True))
            paired.append((task, rival, *(f"{v:+.7f}" for v in delta), f"{avg:+.7f}",
                           f"{center:+.7f}", f"{covariance:+.7f}"))
            if rival == "Flat MLP":
                verdict.append((task, avg, center, covariance, sum(v < 0 for v in delta)))
    for name in models:
        runs = [r for r in all_rows if r["model"] == name]
        time_kind = "per fit" if name == MODEL else "archive, GPU-group amortized"
        runtime.append((name, runs[0]["device"], time_kind, f"{mean(r['training_seconds'] for r in runs):.2f}"))
    display(Markdown("### Primary held-out results"))
    _table(("Task", "Model", "Parameters", "Test W₂²", "Center", "Covariance"), main)
    display(Markdown("### Paired W₂² differences"))
    _table(("Task", "Comparator", *(f"Seed {s}" for s in seeds), "Mean Δ", "Center Δ", "Covariance Δ"), paired)
    display(Markdown("### Distribution shift"))
    _table(("Task", "Model", "ID redraw", "Mean shift", "Covariance shift", "Both shifts"), shifted)
    display(Markdown("### Timing"))
    _table(("Model", "Recorded device", "Timing basis", "Mean seconds"), runtime)
    display(Markdown("Archived GPU timings are amortized across a batch of independent runs. The new arm was timed "
                     "per fit on its recorded device. These figures describe the executions separately."))
    lines = []
    for task, delta, center, covariance, wins in verdict:
        component = "center" if abs(center) >= abs(covariance) else "covariance"
        larger = center if component == "center" else covariance
        label = "lower" if delta < 0 else "higher" if delta > 0 else "equal"
        lines.append(f"- **{task.capitalize()}** has {label} mean W₂² versus Flat MLP "
                     f"({delta:+.7f}, {wins}/3 splits favor MGS-KAN). The larger component difference "
                     f"is {component} ({larger:+.7f}).")
    largest = max(verdict, key=lambda r: abs(r[1]))
    display(Markdown("### Verdict\n\n" + "\n".join(lines) +
                     f"\n\nThe largest absolute Flat MLP gap is on **{largest[0]}**. "
                     "All 162 historical and 27 new fits are paired on the original evaluation seeds. "
                     "This is a paired historical comparison on known seeds. Confirmation on previously unseen "
                     "seeds is left for a later experiment. The individual-fit table contains unrounded W₂² "
                     "and its components by split."))
    print(f"Saved individual-fit rows to {study.output / 'individual_fits.csv'}")
