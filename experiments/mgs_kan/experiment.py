"""Run the frozen comparison or resume a study in the same output directory."""

import argparse
import csv
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from . import config as c
from .checks import run_checks
from .data import make_data
from .models import make_model, parameter_count

DEFAULT_OUTPUT = Path("outputs/experiments/mgs_kan_study")


def write_json(path, payload):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")
    Path(file.name).replace(path)


def read_json(path):
    return json.loads(path.read_text())


def rmse(prediction, target):
    return (prediction - target).square().mean().sqrt().item()


def fit(name, train, validation, init_seed, lr, regularization, epochs=c.EPOCHS, patience=c.PATIENCE):
    start = perf_counter()
    torch.manual_seed(init_seed)
    order_rng = torch.Generator().manual_seed(init_seed)
    model = make_model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=regularization if name == "mlp" else 0.0)
    x, y, _ = train
    val_x, val_y, _ = validation
    best_rmse = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(y), generator=order_rng).split(c.BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[batch]) - y[batch]).square().mean()
            if name == "mgs_kan":
                loss = loss + regularization * model.group_penalty()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            error = rmse(model(val_x), val_y)
        if not np.isfinite(error):
            raise RuntimeError(f"Nonfinite validation RMSE for {name}, epoch {epoch}")
        if error < best_rmse:
            best_rmse, best_epoch, best_state = error, epoch, deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, {"validation_rmse": best_rmse, "best_epoch": best_epoch, "epochs_run": epoch,
                   "training_seconds": perf_counter() - start, "parameters": parameter_count(model)}


def pilot():
    results = []
    for dataset in c.DATASETS:
        (train,), _, _ = make_data(dataset, c.PILOT_DATA_SEED, (c.PILOT_SIZE,), noise_sd=0)
        for model in c.MODELS:
            # The same sample is used for fitting and checkpointing in this capacity check.
            _, metrics = fit(model, train, train, c.PILOT_INIT_SEED, 0.01, 0.0,
                             epochs=c.PILOT_EPOCHS, patience=c.PILOT_EPOCHS)
            assert metrics["validation_rmse"] < c.PILOT_MAX_RMSE, (dataset, model, metrics)
            results.append({"dataset": dataset, "model": model, "training_rmse": metrics["validation_rmse"]})
    return results


def tune():
    data = {dataset: make_data(dataset, c.TUNE_DATA_SEED, c.SIZES[:2])[0] for dataset in c.DATASETS}
    trials, chosen = [], {}
    for model in c.MODELS:
        candidates = []
        for lr, regularization in product(c.LEARNING_RATES, c.REGULARIZATION[model]):
            metrics = {}
            for dataset, (train, val) in data.items():
                _, metrics[dataset] = fit(model, train, val, c.TUNE_INIT_SEED, lr, regularization)
            score = float(np.mean([m["validation_rmse"] for m in metrics.values()]))
            params = {"lr": lr, "regularization": regularization}
            candidates.append((score, params))
            trials.append({"model": model, **params, "mean_validation_rmse": score, "datasets": metrics})
            print(f"Tuning {model:7s} lr={lr:g} regularization={regularization:g} mean RMSE={score:.4f}", flush=True)
        chosen[model] = min(candidates, key=lambda candidate: candidate[0])[1]
    return {"selected": chosen, "trials": trials}


@torch.no_grad()
def pair_diagnostics(model, val_x, population_mean, population_sd, threshold, dataset):
    scores = model.pure_interactions(val_x.double(), population_mean, population_sd).square().mean(dim=0).sqrt()
    ranking = scores.argsort(descending=True, stable=True).tolist()
    planted = set(c.PLANTED) if dataset == "sparse" else set()
    ranked = []
    for rank, index in enumerate(ranking, start=1):
        pair = tuple(model.pairs[index].tolist())
        score = scores[index].item()
        ranked.append({"pair": pair, "rank": rank, "rms": score, "planted": pair in planted,
                       "selected": score > threshold})
    return {"selection_threshold": threshold, "pairs": ranked,
            "top_three_hits": sum(pair["planted"] for pair in ranked[:3]),
            "selected_pairs": sum(pair["selected"] for pair in ranked),
            "false_selections": sum(pair["selected"] and not pair["planted"] for pair in ranked)}


def final_fit(path, model_name, dataset, data_seed, init_seed, splits, mean, sd, selected):
    train, val, test = splits
    model, metrics = fit(model_name, train, val, init_seed, **selected)
    with torch.no_grad():
        prediction = model(test[0])
    info = {"dataset": dataset, "data_seed": data_seed, "init_seed": init_seed, "model": model_name,
            "settings": selected, **metrics, "test_rmse": rmse(prediction, test[1])}
    with TemporaryDirectory(dir=path.parent, prefix=".fit-") as temporary:
        staging = Path(temporary)
        torch.save(model.state_dict(), staging / "checkpoint.pt")
        np.savez_compressed(staging / "test_predictions.npz", prediction=prediction.numpy(),
                            target=test[1].numpy(), clean_target=test[2].numpy())
        if model_name == "mgs_kan":
            threshold = c.SELECTION_FRACTION * train[1].std(correction=0).item()
            pairs = pair_diagnostics(model.double(), val[0], mean, sd, threshold, dataset)
            write_json(staging / "pairs.json", pairs)
            info.update({key: value for key, value in pairs.items() if key != "pairs"})
        write_json(staging / "metrics.json", info)
        staging.rename(path)
    return info


def report_rows(runs):
    rows = []
    for dataset, seed in product(c.DATASETS, c.DATA_SEEDS):
        group = [r for r in runs if r["dataset"] == dataset and r["data_seed"] == seed]
        row = {"dataset": dataset, "data_seed": seed}
        for model in c.MODELS:
            fits = [r for r in group if r["model"] == model]
            assert {r["init_seed"] for r in fits} == set(c.INIT_SEEDS) and len(fits) == len(c.INIT_SEEDS)
            row[f"{model}_rmse"] = float(np.mean([fit["test_rmse"] for fit in fits]))
        row["mgs_minus_mlp"] = row["mgs_kan_rmse"] - row["mlp_rmse"]
        rows.append(row)
    return rows


def run_study(output=DEFAULT_OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(c.THREADS)
    torch.use_deterministic_algorithms(True)
    manifest = {"config": {key: value for key, value in vars(c).items() if key.isupper()},
                "code_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(__file__).parent.glob("*.py"))},
                "torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
                "platform": platform.platform(), "device": "cpu"}
    manifest_path = output / "protocol.json"
    if manifest_path.exists():
        if read_json(manifest_path) != json.loads(json.dumps(manifest)):
            raise ValueError(f"Protocol changed. Choose a new output directory instead of mixing runs in {output}.")
    else:
        write_json(manifest_path, manifest)
    write_json(output / "checks.json", run_checks())
    print("Float64 checks passed.", flush=True)
    pilot_path = output / "pilot.json"
    if not pilot_path.exists():
        write_json(pilot_path, pilot())
        print("Noiseless pilot passed for both models on both datasets.", flush=True)
    tuning_path = output / "tuning.json"
    if not tuning_path.exists():
        write_json(tuning_path, tune())
    chosen = read_json(tuning_path)["selected"]
    print(f"Frozen training settings: {chosen}", flush=True)

    runs = []
    fits = output / "fits"
    fits.mkdir(exist_ok=True)
    for dataset, data_seed in product(c.DATASETS, c.DATA_SEEDS):
        splits, mean, sd = make_data(dataset, data_seed)
        for init_seed, model in product(c.INIT_SEEDS, c.MODELS):
            path = fits / f"{dataset}-{data_seed}-{init_seed}-{model}"
            if path.exists():
                info = read_json(path / "metrics.json")
            else:
                info = final_fit(path, model, dataset, data_seed, init_seed, splits, mean, sd, chosen[model])
            runs.append(info)
            print(f"Fit {len(runs):2d}/36 {path.name} checkpoint epoch {info['best_epoch']}", flush=True)
    rows = report_rows(runs)
    for filename, records in (("runs.csv", runs), ("paired_rmse.csv", rows)):
        fields = list(dict.fromkeys(key for record in records for key in record))
        with (output / filename).open("w", newline="") as file:
            csv_writer = csv.DictWriter(file, fieldnames=fields)
            csv_writer.writeheader()
            csv_writer.writerows(records)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_study(parser.parse_args().output)
