"""Numerical checks, training, verified checkpoints and paired evaluation."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from itertools import combinations, product
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import platform
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import torch
from torch.func import functional_call

from . import config as c
from .data import estimate_reference, make_data, planted_pairs
from .models import MGS_KAN, make_model, parameter_count

DEFAULT_OUTPUT = Path("outputs/study")
ORIGINAL_DIRECTORY = "df24c2b5a1de3a48"
ORIGINAL_MANIFEST_HASH = "2b5dca1569b68075876e94f5a39878ee22bc676fc15b9ea98f9f499e170ff210"


@contextmanager
def _atomic_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        yield temp
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _write_json(path, value):
    with _atomic_file(path) as temp:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _setup_torch():
    torch.set_num_threads(c.THREADS)
    torch.use_deterministic_algorithms(True)


def _runtime():
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {"torch": str(torch.__version__), "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "device": c.DEVICE, "cpu": cpu}


def original_archive(root=DEFAULT_OUTPUT):
    """Open the original evidence read-only on the recorded runtime."""
    output = Path(root) / ORIGINAL_DIRECTORY
    manifest = output / "final" / "manifest.json"
    if not manifest.exists():
        return None
    assert sha256(manifest.read_bytes()).hexdigest() == ORIGINAL_MANIFEST_HASH
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["id"] == _digest({k: v for k, v in protocol.items() if k != "id"})
    assert protocol["id"] == json.loads(manifest.read_text())["protocol_id"]
    if protocol["settings"]["runtime"] != _runtime():
        return None
    _setup_torch()
    return output, protocol


def _original_design():
    return {"tasks": c.TASKS, "data_seeds": c.DATA_SEEDS, "tune_tasks": c.TASKS,
            "development_seed": c.DEV_DATA_SEED, "inherited_settings": {}, "workers": 1}


def _design(protocol):
    # The archived first study predates the explicit design record.
    return protocol["settings"].get("design", _original_design())


def freeze_protocol(root=DEFAULT_OUTPUT, *, inherited_settings=None):
    _setup_torch()
    design = _original_design()
    if inherited_settings is not None:
        assert set(inherited_settings) == set(c.TASKS)
        design = {"tasks": c.FOLLOWUP_TASKS, "data_seeds": c.FOLLOWUP_DATA_SEEDS,
                  "tune_tasks": ("sparse_overlap",), "development_seed": c.FOLLOWUP_DEV_SEED,
                  "inherited_settings": inherited_settings, "workers": c.FOLLOWUP_WORKERS}
        assert not set(c.FOLLOWUP_DATA_SEEDS) & {c.DEV_DATA_SEED, *c.DATA_SEEDS}
        assert c.FOLLOWUP_DEV_SEED not in (c.PILOT_DATA_SEED, c.DEV_DATA_SEED, *c.DATA_SEEDS)
    seeds = design["data_seeds"]
    assert len({c.PILOT_DATA_SEED, design["development_seed"], *seeds}) == len(seeds) + 2
    assert len({c.PILOT_INIT_SEED, c.DEV_INIT_SEED, *c.INIT_SEEDS}) == 5
    settings = {
        "config": {key: value for key, value in vars(c).items() if key.isupper()},
        "code_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                        for p in sorted(Path(__file__).parent.glob("*.py"))},
        "runtime": _runtime(),
        "design": design,
    }
    settings = json.loads(json.dumps(settings))
    output = Path(root) / _digest(settings)[:16]
    path = output / "protocol.json"
    if path.exists():
        protocol = json.loads(path.read_text())
        assert protocol["settings"] == settings
        assert protocol["id"] == _digest({k: v for k, v in protocol.items() if k != "id"})
    else:
        protocol = {"settings": settings, "reference": estimate_reference(design["tasks"])}
        protocol["id"] = _digest(protocol)
        _write_json(path, protocol)
    return output, protocol


def centered_diagonal(function, u, v, chunk=c.SCORE_CHUNK):
    """Double-center over the full Cartesian product; return observed rows."""
    rows = []
    columns = torch.zeros_like(v)
    for u_chunk in u.split(chunk):
        grid = function(u_chunk[:, None], v[None, :])
        rows.append(grid.mean(dim=1))
        columns += grid.sum(dim=0)
    columns /= len(u)
    return function(u, v) - torch.cat(rows) - columns + columns.mean()


def numerical_checks(output):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c.CHECK_SEED)
        model = MGS_KAN().double()
        x = torch.randn(3, c.FEATURES, dtype=torch.float64, requires_grad=True)
        counts = {name: parameter_count(make_model(name)) for name in c.MODELS}
    gap = abs(counts["mgs_kan"] - counts["mlp"]) / counts["mgs_kan"]
    assert counts == {"mgs_kan": 949, "mlp": 937} and gap <= c.MAX_PARAMETER_GAP
    pairs = list(combinations(range(c.FEATURES), 2))
    assert model.pairs.tolist() == [list(pair) for pair in pairs] and len(pairs) == 66
    affine, singles, terms = model.contributions(x)
    torch.testing.assert_close(model(x), affine + singles.sum(1) + terms.sum(1), rtol=0, atol=0)
    precision = model.precision()
    torch.testing.assert_close(precision, precision.mT, rtol=0, atol=1e-15)
    min_eigenvalue = torch.linalg.eigvalsh(precision).min().item()
    assert min_eigenvalue > 0 and torch.all(model.single_log_scale.exp().square() > 0)

    names, parameters = zip(*model.named_parameters())
    assert torch.autograd.gradcheck(
        lambda inputs, *params: functional_call(model, dict(zip(names, params)), (inputs,)),
        (x, *parameters),
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        cosine, sine = np.cos(c.GAUSSIAN_ANGLE), np.sin(c.GAUSSIAN_ANGLE)
        rotation = torch.tensor([[cosine, -sine], [sine, cosine]], dtype=x.dtype)
        eigenvalues = torch.tensor(c.GAUSSIAN_EIGENVALUES, dtype=x.dtype)
        lower = torch.linalg.cholesky(rotation @ eigenvalues.diag() @ rotation.T)
        index = 4  # A single nonzero kernel at any candidate pair suffices.
        model.pair_center[index, 0] = torch.tensor(c.GAUSSIAN_CENTER, dtype=x.dtype)
        model.pair_amplitude[index, 0] = c.GAUSSIAN_AMPLITUDE
        model.pair_precision_raw[index, 0] = torch.stack(
            (lower[0, 0].log(), lower[1, 0], lower[1, 1].log())
        )
        u = torch.tensor([0., 1., -1., 0.4, -0.3, 1.2], dtype=x.dtype)
        v = torch.tensor([0., 1., 1., -0.8, 0.8, -0.7], dtype=x.dtype)
        probes = x.new_zeros(len(u), c.FEATURES)
        probes[:, pairs[index][0]] = c.GAUSSIAN_CENTER[0] + u
        probes[:, pairs[index][1]] = c.GAUSSIAN_CENTER[1] + v
        p, q = cosine * u + sine * v, -sine * u + cosine * v
        expected = c.GAUSSIAN_AMPLITUDE * torch.exp(
            -0.5 * (eigenvalues[0] * p.square() + eigenvalues[1] * q.square())
        )
        gaussian_error = (model(probes) - expected).abs().max().item()
        assert gaussian_error < 1e-10

    additive = centered_diagonal(lambda a, b: a.square() + b.sin() + 0.7, u, v, chunk=2)
    centered_product = centered_diagonal(torch.mul, u, v, chunk=2)
    expected_product = (u - u.mean()) * (v - v.mean())
    additive_error = additive.abs().max().item()
    product_error = (centered_product - expected_product).abs().max().item()
    assert max(additive_error, product_error) < 1e-12
    result = {"status": "passed", "dtype": "float64", "candidate_pairs": len(pairs),
              "parameter_counts": counts, "relative_parameter_gap": gap,
              "minimum_precision_eigenvalue": min_eigenvalue, "gradcheck": "passed",
              "substitute_gaussian_max_abs_error": gaussian_error,
              "additive_max_abs_error": additive_error, "product_max_abs_error": product_error}
    _write_json(output / "checks.json", result)
    return result


def rmse(prediction, target):
    return (prediction.double() - target.double()).square().mean().sqrt().item()


def _initialized_model(name, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return make_model(name).to(c.DEVICE)


def _fit(name, train, validation, init_seed, settings, epochs=c.MAX_EPOCHS, patience=c.PATIENCE):
    model = _initialized_model(name, init_seed)
    batch_seed = int(np.random.SeedSequence([c.BATCH_STREAM, init_seed]).generate_state(1)[0])
    order_rng = torch.Generator().manual_seed(batch_seed)
    lr, penalty = settings["lr"], settings["regularization"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=penalty if name == "mlp" else 0.)
    x, y = (tensor.to(c.DEVICE) for tensor in train)
    val_x, val_y = (tensor.to(c.DEVICE) for tensor in validation)
    history = []
    best_rmse = float("inf")
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(y), generator=order_rng).split(c.BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[batch]) - y[batch]).square().mean()
            if name == "mgs_kan" and penalty:
                loss = loss + penalty * model.group_penalty()
            loss.backward()
            optimizer.step()
        if epoch % c.VALIDATE_EVERY:
            continue
        model.eval()
        with torch.no_grad():
            error = rmse(model(val_x), val_y)
        if not np.isfinite(error):
            raise RuntimeError(f"Nonfinite validation RMSE for {name} at epoch {epoch}")
        history.append(error)
        if error < best_rmse:
            best_rmse, best_epoch = error, epoch
            best_state = deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, {"validation_rmse": best_rmse, "best_epoch": best_epoch, "epochs_run": epoch,
                   "training_seconds": perf_counter() - start, "validation_history": history,
                   "parameters": parameter_count(model), "batch_seed": batch_seed}


def run_pilot(output, protocol):
    results = []
    for task in _design(protocol)["tune_tasks"]:
        (train,), _ = make_data(task, c.PILOT_DATA_SEED, protocol["reference"], (c.PILOT_SIZE,), noise_sd=0.)
        for name in c.MODELS:
            settings = {"lr": c.PILOT_LR, "regularization": 0.}
            _, metrics = _fit(name, train, train, c.PILOT_INIT_SEED, settings, c.PILOT_EPOCHS, c.PILOT_EPOCHS)
            results.append({"task": task, "model": name, "training_rmse": metrics["validation_rmse"],
                            "epoch": metrics["best_epoch"], "training_seconds": metrics["training_seconds"]})
    _write_json(output / "pilot" / "results.json", results)
    failures = [r for r in results if r["training_rmse"] >= c.PILOT_MAX_RMSE]
    if failures:
        raise RuntimeError(f"Noiseless training pilot missed the RMSE target: {failures}")
    return results


@torch.no_grad()
def _pair_scores(model, x, threshold, task):
    model = deepcopy(model).double()
    x = x.to(c.DEVICE, torch.float64)
    planted = planted_pairs(task)
    rows = []
    for index, pair in enumerate(map(tuple, model.pairs.tolist())):
        u, v = x[:, pair].unbind(dim=1)
        centered = centered_diagonal(lambda a, b: model.pair_values(a, b, index), u, v)
        score = centered.square().mean().sqrt().item()
        rows.append({"pair": pair, "score": score, "selected": score > threshold, "planted": pair in planted})
    rows.sort(key=lambda row: (-row["score"], row["pair"]))
    return [{"rank": rank, **row} for rank, row in enumerate(rows, start=1)]


def _identity(protocol, task, seed, init, name, settings):
    return {"protocol_id": protocol["id"], "task": task, "data_seed": seed,
            "init_seed": init, "model": name, "device": c.DEVICE, "settings": settings}


def _fit_key(identity):
    return "{task}-{data_seed}-{init_seed}-{model}".format(**identity)


def _threshold(train):
    return c.SELECTION_FRACTION * train[1].double().std(correction=0).item()


@torch.no_grad()
def _load_verified(path, identity, train, validation, test=None):
    run = torch.load(path, map_location=c.DEVICE, weights_only=True)
    assert run["identity"] == identity
    model = _initialized_model(identity["model"], identity["init_seed"]).eval()
    model.load_state_dict(run["checkpoint"], strict=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    metrics = run["metrics"]
    history = metrics["validation_history"]
    best = min(range(len(history)), key=history.__getitem__)
    assert np.isfinite(history).all() and np.isfinite(metrics["training_seconds"])
    assert metrics["training_seconds"] >= 0
    assert metrics["best_epoch"] == (best + 1) * c.VALIDATE_EVERY
    assert metrics["epochs_run"] // c.VALIDATE_EVERY == len(history)
    assert metrics["epochs_run"] <= c.MAX_EPOCHS
    assert metrics["epochs_run"] == c.MAX_EPOCHS or metrics["epochs_run"] - metrics["best_epoch"] >= c.PATIENCE
    assert metrics["parameters"] == parameter_count(model)
    val_x, val_y = (tensor.to(c.DEVICE) for tensor in validation)
    assert rmse(model(val_x), val_y) == metrics["validation_rmse"] == history[best]

    if test is not None:
        x, target = (tensor.to(c.DEVICE) for tensor in test)
        held_out = run["test"]
        offset = sum(c.SPLIT_SIZES[:2])
        assert torch.equal(held_out["rows"], torch.arange(offset, offset + len(target), device=c.DEVICE))
        assert torch.equal(held_out["target"], target)
        assert torch.equal(model(x), held_out["prediction"])
        assert np.isfinite(metrics["test_rmse"])
        assert rmse(held_out["prediction"], held_out["target"]) == metrics["test_rmse"]
        if identity["model"] == "mgs_kan":
            pairs = run["pairs"]
            threshold = _threshold(train)
            assert torch.equal(model.pairs.cpu(), torch.triu_indices(c.FEATURES, c.FEATURES, offset=1).T)
            assert metrics["selection_threshold"] == threshold
            assert sorted(row["pair"] for row in pairs) == list(combinations(range(c.FEATURES), 2))
            assert pairs == sorted(pairs, key=lambda row: (-row["score"], row["pair"]))
            for rank, row in enumerate(pairs, start=1):
                assert row["rank"] == rank and np.isfinite(row["score"]) and row["score"] >= 0
                assert row["selected"] == (row["score"] > threshold)
                assert row["planted"] == (row["pair"] in planted_pairs(identity["task"]))
    return run


def _run_or_resume(path, identity, train, validation, test=None):
    try:
        return _load_verified(path, identity, train, validation, test), True
    except (FileNotFoundError, AssertionError, EOFError, RuntimeError, pickle.UnpicklingError,
            KeyError, TypeError, ValueError, IndexError) as error:
        if path.exists():
            print(f"Rebuilding {path.name} after failed verification ({type(error).__name__}).", flush=True)

    model, metrics = _fit(identity["model"], train, validation, identity["init_seed"], identity["settings"])
    run = {"identity": identity, "metrics": metrics, "checkpoint": model.state_dict()}
    if test is not None:
        x, target = (tensor.to(c.DEVICE) for tensor in test)
        with torch.no_grad():
            prediction = model(x)
        offset = sum(c.SPLIT_SIZES[:2])
        run["test"] = {"rows": torch.arange(offset, offset + len(target)),
                       "target": target.clone(), "prediction": prediction}
        metrics["test_rmse"] = rmse(prediction, target)
        if identity["model"] == "mgs_kan":
            metrics["selection_threshold"] = _threshold(train)
            run["pairs"] = _pair_scores(model, validation[0], metrics["selection_threshold"], identity["task"])

    with _atomic_file(path) as temp:
        torch.save(run, temp)
        _load_verified(temp, identity, train, validation, test)
    return run, False


def _summary(run):
    return {**run["identity"], **{k: v for k, v in run["metrics"].items() if k != "validation_history"}}


def tune(output, protocol):
    design = _design(protocol)
    trials, selected = [], dict(design["inherited_settings"])
    seed = design["development_seed"]
    for task in design["tune_tasks"]:
        (train, validation), _ = make_data(task, seed, protocol["reference"], c.SPLIT_SIZES[:2])
        selected[task] = {}
        for name in c.MODELS:
            candidates = []
            for trial, (lr, penalty) in enumerate(product(c.LEARNING_RATES, c.REGULARIZATION[name]), start=1):
                settings = {"lr": lr, "regularization": penalty}
                identity = _identity(protocol, task, seed, c.DEV_INIT_SEED, name, settings)
                path = output / "tuning" / f"{task}-{name}-{trial}.pt"
                run, reused = _run_or_resume(path, identity, train, validation)
                row = {"trial": trial, **_summary(run)}
                candidates.append(row)
                print(f"{task} {name} trial {trial} • validation RMSE {row['validation_rmse']:.5f}"
                      f"{' (verified)' if reused else ''}", flush=True)
            winner = min(candidates, key=lambda row: row["validation_rmse"])
            selected[task][name] = winner["settings"]
            trials.extend({**row, "selected": row is winner} for row in candidates)
    tuning = {"selected": selected, "trials": trials}
    path = output / "tuning" / "selection.json"
    if not path.exists() or json.loads(path.read_text()) != tuning:
        _write_json(path, tuning)
    return tuning


def _split_jobs(output, protocol, selected, task, seed):
    splits, diagnostics = make_data(task, seed, protocol["reference"])
    for init, name in product(c.INIT_SEEDS, c.MODELS):
        identity = _identity(protocol, task, seed, init, name, selected[task][name])
        path = output / "final" / f"{_fit_key(identity)}.pt"
        yield path, identity, splits, diagnostics


def _run_split(output, protocol, selected, task, seed):
    reused = sum(_run_or_resume(path, identity, *splits)[1]
                 for path, identity, splits, _ in _split_jobs(output, protocol, selected, task, seed))
    return task, seed, reused


def run_final(output, protocol, selected):
    frozen = json.loads((output / "tuning" / "selection.json").read_text())
    assert frozen["selected"] == selected
    design = _design(protocol)
    groups = list(product(design["tasks"], design["data_seeds"]))
    group_size = len(c.INIT_SEEDS) * len(c.MODELS)
    total = len(groups) * group_size

    def report(count, result):
        task, seed, reused = result
        print(f"{count * group_size}/{total} fits complete • {task}, seed {seed} "
              f"({reused}/{group_size} reused and verified)", flush=True)

    if design["workers"] == 1:
        for count, (task, seed) in enumerate(groups, start=1):
            report(count, _run_split(output, protocol, selected, task, seed))
    else:
        with ProcessPoolExecutor(max_workers=design["workers"], mp_context=get_context("spawn"),
                                 initializer=_setup_torch) as pool:
            jobs = [pool.submit(_run_split, output, protocol, selected, task, seed) for task, seed in groups]
            for count, job in enumerate(as_completed(jobs), start=1):
                report(count, job.result())


def verify_final(output, protocol, *, archived=False):
    """Reload the complete study from disk. Raise before reporting a partial study."""
    selected = json.loads((output / "tuning" / "selection.json").read_text())["selected"]
    design = _design(protocol)
    jobs = [job for task, seed in product(design["tasks"], design["data_seeds"])
            for job in _split_jobs(output, protocol, selected, task, seed)]
    expected = {path.stem for path, *_ in jobs}
    assert len(jobs) == len(expected) == len(design["tasks"]) * len(design["data_seeds"]) * len(c.INIT_SEEDS) * len(c.MODELS)
    actual = {path.stem for path in (output / "final").glob("*.pt")}
    if actual != expected:
        raise RuntimeError(f"Final fits missing: {sorted(expected - actual)}. Unexpected: {sorted(actual - expected)}")
    fits, recovery, variances = {}, [], {}
    for path, identity, splits, diagnostics in jobs:
        run = _load_verified(path, identity, *splits)
        row = _summary(run)
        fits[path.stem] = {**row, "artifact": str(path.relative_to(output)),
                          "sha256": sha256(path.read_bytes()).hexdigest()}
        task, seed = row["task"], row["data_seed"]
        variances[task, seed] = [{"task": task, "data_seed": seed, **r} for r in diagnostics]
        if identity["model"] == "mgs_kan":
            pairs = run["pairs"]
            ranks = {pair: next(p["rank"] for p in pairs if p["pair"] == pair) for pair in planted_pairs(task)}
            recovery.append({"task": task, "data_seed": seed, "init_seed": row["init_seed"],
                             "ranks": ranks, "top_three_hits": sum(rank <= 3 for rank in ranks.values()),
                             "selected": sum(p["selected"] for p in pairs),
                             "selected_planted": sum(p["selected"] and p["planted"] for p in pairs),
                             "false_selections": sum(p["selected"] and not p["planted"] for p in pairs),
                             "threshold": row["selection_threshold"]})
    manifest = {"protocol_id": protocol["id"], "fits": fits}
    path = output / "final" / "manifest.json"
    if archived:
        assert json.loads(path.read_text()) == manifest
    else:
        _write_json(path, manifest)
    return list(fits.values()), recovery, [row for group in variances.values() for row in group]


def paired_results(fits, tasks=c.TASKS, seeds=c.DATA_SEEDS):
    rows = []
    for task, seed in product(tasks, seeds):
        means = {}
        for name in c.MODELS:
            group = [r for r in fits if (r["task"], r["data_seed"], r["model"]) == (task, seed, name)]
            assert sorted(r["init_seed"] for r in group) == sorted(c.INIT_SEEDS)
            means[name] = float(np.mean([r["test_rmse"] for r in group]))
        rows.append({"task": task, "data_seed": seed, **means, "difference": means["mgs_kan"] - means["mlp"]})
    return rows
