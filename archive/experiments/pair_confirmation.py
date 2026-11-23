"""Fresh-split confirmation of the frozen pair expansion."""

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import product
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments import pair_ablation as ablation
from src import config as c
from src.data import make_data
from src.experiment import _atomic_file, _digest, _fit, _initialized_model, _runtime, _setup_torch, _write_json, rmse
from src.models import parameter_count

DATA_SEEDS = tuple(range(4101, 4111))
SETTINGS = {
    "mgs_8": {"lr": 0.003, "regularization": 0.001},
    "mgs_kan": {"lr": 0.001, "regularization": 0.001},
    "mlp": {"lr": 0.003, "regularization": 0.},
}
COUNTS = {"mgs_8": 985, "mgs_kan": 949, "mlp": 937}


def confirmation_model(name, init):
    return ablation._model(8, init) if name == "mgs_8" else _initialized_model(name, init)


def _train(name, train, validation, init):
    settings = SETTINGS[name]
    if name != "mgs_8":
        return _fit(name, train, validation, init, settings)
    model = confirmation_model(name, init)
    checkpoints, training = ablation._fit(model, train, validation, init, settings["lr"], settings["regularization"])
    model.load_state_dict(checkpoints["selected"])
    best = training.pop("selected_epoch")
    stop = training.pop("stop_epoch")
    return model, {
        **training, "best_epoch": best, "epochs_run": stop,
        "parameters": parameter_count(model), "validation_rmse": training["validation_history"][best - 1],
    }


@torch.inference_mode()
def _verify(path, identity, splits):
    fit = torch.load(path, weights_only=True, map_location="cpu")
    assert fit["identity"] == identity
    name, init = identity["model"], identity["init"]
    model = confirmation_model(name, init).eval()
    model.load_state_dict(fit["checkpoint"], strict=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    metrics = fit["metrics"]
    history = metrics["validation_history"]
    stop = metrics["epochs_run"]
    best = min(range(stop), key=history.__getitem__) + 1
    assert 0 < stop <= len(history) <= c.MAX_EPOCHS and np.isfinite(history).all()
    assert best == metrics["best_epoch"] and stop == min(best + c.PATIENCE, c.MAX_EPOCHS)
    assert np.isfinite(metrics["training_seconds"]) and metrics["training_seconds"] >= 0
    assert parameter_count(model) == COUNTS[name] == metrics["parameters"]
    batch_seed = int(np.random.SeedSequence([c.BATCH_STREAM, init]).generate_state(1)[0])
    assert metrics["batch_seed"] == batch_seed
    _, (val_x, val_y), (test_x, test_y) = splits
    assert rmse(model(val_x), val_y) == history[best - 1] == metrics["validation_rmse"]
    assert torch.equal(fit["target"], test_y)
    offset = sum(c.SPLIT_SIZES[:2])
    assert torch.equal(fit["rows"], torch.arange(offset, offset + len(test_y)))
    assert torch.equal(model(test_x), fit["prediction"])
    assert rmse(fit["prediction"], test_y) == metrics["test_rmse"]
    return fit


def _job(args):
    output, protocol_id, reference, seed, init, name = args
    key = f"{seed}-{init}-{name}"
    path = output / "fits" / f"{key}.pt"
    identity = {"protocol_id": protocol_id, "seed": seed, "init": init,
                "model": name, "settings": SETTINGS[name]}
    splits, _ = make_data(ablation.TASK, seed, reference)
    reused = path.exists()
    if reused:
        result = _verify(path, identity, splits)
    else:
        train, validation, (test_x, target) = splits
        model, metrics = _train(name, train, validation, init)
        with torch.inference_mode():
            prediction = model(test_x)
        metrics["test_rmse"] = rmse(prediction, target)
        offset = sum(c.SPLIT_SIZES[:2])
        result = {"identity": identity, "checkpoint": model.state_dict(), "metrics": metrics,
                  "target": target, "rows": torch.arange(offset, offset + len(target)),
                  "prediction": prediction}
        with _atomic_file(path) as temp:
            torch.save(result, temp)
            _verify(temp, identity, splits)
    summary = {k: v for k, v in result["metrics"].items() if k != "validation_history"}
    row = {**identity, **summary, "artifact": str(path.relative_to(output)),
           "sha256": sha256(path.read_bytes()).hexdigest()}
    return key, row, reused


def run_confirmation(root, follow, follow_protocol, ablation_output):
    _setup_torch()
    ablation._check_initialization()
    assert c.DEVICE == "cpu" and c.VALIDATE_EVERY == 1
    prior_seeds = {*c.DATA_SEEDS, *c.FOLLOWUP_DATA_SEEDS, *ablation.SEEDS,
                   c.DEV_DATA_SEED, c.FOLLOWUP_DEV_SEED, c.PILOT_DATA_SEED, c.REFERENCE_SEED}
    assert not prior_seeds.intersection(DATA_SEEDS)
    development = json.loads((follow / "tuning" / "selection.json").read_text())["selected"][ablation.TASK]
    assert development == {name: SETTINGS[name] for name in c.MODELS}
    upstream_protocol = json.loads((ablation_output / "protocol.json").read_text())
    upstream_manifest = json.loads((ablation_output / "manifest.json").read_text())
    upstream_id = _digest(upstream_protocol)
    assert upstream_id == upstream_manifest["protocol_id"]
    assert upstream_protocol["upstream_id"] == follow_protocol["id"]
    expanded = SETTINGS["mgs_8"]
    assert any((row["kernels"], row["lr"], row["lambda"]) == (8, expanded["lr"], expanded["regularization"])
               for row in upstream_manifest["rows"])
    protocol = {
        "task": ablation.TASK, "expanded_pair": ablation.PAIR, "expanded_kernels": 8,
        "data_seeds": DATA_SEEDS, "init_seeds": c.INIT_SEEDS, "settings": SETTINGS,
        "parameters": COUNTS, "split_sizes": c.SPLIT_SIZES, "epochs": c.MAX_EPOCHS,
        "patience": c.PATIENCE, "batch_size": c.BATCH_SIZE, "runtime": _runtime(),
        "workers": c.FOLLOWUP_WORKERS, "threads": c.THREADS,
        "pair_error": {"normal_quadrature_nodes": 40, "mlp_backgrounds": 256, "sobol_seed": 5317},
        "follow_protocol_id": follow_protocol["id"], "ablation_protocol_id": upstream_id,
        "code_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), Path(ablation.__file__))},
    }
    protocol = json.loads(json.dumps(protocol))
    protocol_id = _digest(protocol)
    output = Path(root) / protocol_id[:16]
    path = output / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        _write_json(path, protocol)

    jobs = [(output, protocol_id, follow_protocol["reference"], *row)
            for row in product(DATA_SEEDS, c.INIT_SEEDS, SETTINGS)]
    fits, reused = {}, 0
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=c.FOLLOWUP_WORKERS, mp_context=get_context("spawn"),
                             initializer=_setup_torch) as pool:
        for i, (key, row, was_reused) in enumerate(pool.map(_job, jobs), start=1):
            fits[key] = row
            reused += was_reused
            if i % 9 == 0:
                print(f"{i}/{len(jobs)} fits verified ({reused} reused)", flush=True)
    assert len(fits) == 90 == len(jobs)
    assert {p.stem for p in (output / "fits").glob("*.pt")} == set(fits)
    manifest = {"protocol_id": protocol_id, "fits": fits}
    path = output / "manifest.json"
    if not path.exists() or json.loads(path.read_text()) != manifest:
        _write_json(path, manifest)
    return output, manifest, perf_counter() - start
