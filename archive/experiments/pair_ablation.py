"""Read-only extension of the frozen MGS-KAN studies."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src import config as c
from src.data import _components, _overlap_components, make_data
from src.experiment import _atomic_file, _digest, _runtime, _setup_torch, _threshold, _write_json, centered_diagonal, rmse
from src.models import MGS_KAN, parameter_count

TASK, PAIR = "sparse_overlap", (2, 4)
SEEDS, KERNELS, RATES, PENALTIES = (3301, 3307, 3313), (2, 8), (0.001, 0.003), (0.001, 0.)
BASELINE = (2, 0.001, 0.001)
EXTRA_STREAM, EVAL_SEED, EVAL_POWER, QUADRATURE = 6143, 6899, 15, 40
ENDPOINTS = ("selected", "epoch_300")


class ExpandedPair(MGS_KAN):
    """Add Gaussians only to the designated pair and share its group penalty."""

    def __init__(self, kernels, seed):
        super().__init__()
        self.target_index = self.pairs.tolist().index(list(PAIR))
        extra = kernels - c.PAIR_KERNELS
        seed = int(np.random.SeedSequence([EXTRA_STREAM, seed]).generate_state(1)[0])
        rng = torch.Generator().manual_seed(seed)
        self.extra_center = nn.Parameter(c.PAIR_CENTER_SD * torch.randn(extra, 2, generator=rng))
        self.extra_raw = nn.Parameter(c.PAIR_FACTOR_SD * torch.randn(extra, 3, generator=rng))
        self.extra_amplitude = nn.Parameter(torch.zeros(extra))
        mask = torch.arange(len(self.pairs)) == self.target_index
        self.register_buffer("expanded_pair_mask", mask)

    def _extra_values(self, u, v):
        cu, cv = self.extra_center.unbind(dim=-1)
        log_a, b, log_d = self.extra_raw.unbind(dim=-1)
        du, dv = u.unsqueeze(-1) - cu, v.unsqueeze(-1) - cv
        q = (log_a.exp() * du + b * dv).square() + (log_d.exp() * dv).square()
        return (torch.exp(-0.5 * q) * self.extra_amplitude).sum(dim=-1)

    def pair_values(self, u, v, pair=slice(None)):
        value = super().pair_values(u, v, pair)
        if isinstance(pair, int):
            return value + self._extra_values(u, v) if pair == self.target_index else value
        assert pair == slice(None)
        index = self.target_index
        extra = self._extra_values(u[..., index], v[..., index])
        return value + extra.unsqueeze(-1) * self.expanded_pair_mask

    def group_penalty(self):
        index = self.target_index
        before, target, after = self.pair_amplitude[:index], self.pair_amplitude[index], self.pair_amplitude[index + 1:]
        return (torch.linalg.vector_norm(before, dim=1).sum()
                + torch.linalg.vector_norm(torch.cat((target, self.extra_amplitude)))
                + torch.linalg.vector_norm(after, dim=1).sum())


def _model(kernels, init):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init)
        return MGS_KAN() if kernels == c.PAIR_KERNELS else ExpandedPair(kernels, init)


def _check_initialization():
    x = torch.linspace(-2., 2., 17 * c.FEATURES).reshape(17, c.FEATURES)
    for init in c.INIT_SEEDS:
        base, expanded = (_model(k, init) for k in KERNELS)
        for name, value in base.state_dict().items():
            assert torch.equal(value, expanded.state_dict()[name])
        torch.testing.assert_close(base(x), expanded(x), atol=0, rtol=0)
        torch.testing.assert_close(base.group_penalty(), expanded.group_penalty())
        assert parameter_count(expanded) - parameter_count(base) == (KERNELS[1] - KERNELS[0]) * 6
        # A new zero-amplitude kernel can begin learning through its amplitude.
        expanded(x).square().mean().backward()
        assert expanded.extra_amplitude.grad.abs().max() > 0


@torch.inference_mode()
def _evaluate(model, validation, test, mean, scale, reference):
    val_x, val_y = validation
    x, y = test
    prediction = model(x)
    axis, weights = np.polynomial.hermite.hermgauss(QUADRATURE)
    axis = torch.from_numpy(axis * np.sqrt(2))
    weights = torch.from_numpy(weights / np.sqrt(np.pi))
    raw_pair = torch.cartesian_prod(axis, axis)
    raw = axis.new_zeros((len(raw_pair), c.FEATURES))
    raw[:, PAIR] = raw_pair
    overlap = reference["overlap"]
    component = c.OVERLAP_PAIRS.index(PAIR) + len(c.ADDITIVE_FEATURES)
    true_pair = torch.from_numpy(_overlap_components(raw.numpy(), _components(raw.numpy())[:, :3])[:, component])
    true_pair = (true_pair - overlap["component_mean"][component]) / (
        overlap["component_sd"][component] * reference["targets"][TASK]["sd"]
    )
    pair_model = deepcopy(model).double()
    index = pair_model.pairs.tolist().index(list(PAIR))
    i, j = PAIR
    u, v = (axis - mean[i]) / scale[i], (axis - mean[j]) / scale[j]
    error = pair_model.pair_values(u[:, None], v[None, :], index) - true_pair.reshape(QUADRATURE, QUADRATURE)
    pure_error = error - (error @ weights)[:, None] - (weights @ error)[None, :] + weights @ error @ weights

    sampler = torch.quasirandom.SobolEngine(c.FEATURES, scramble=True, seed=EVAL_SEED)
    population = torch.special.ndtri(sampler.draw_base2(EVAL_POWER, dtype=torch.float64).clamp(1e-12, 1 - 1e-12))
    contents = _overlap_components(population.numpy(), _components(population.numpy())[:, :3])
    standardized = (contents - overlap["component_mean"]) / overlap["component_sd"]
    target = reference["targets"][TASK]
    clean_y = torch.from_numpy((standardized.sum(axis=1) - target["mean"]) / target["sd"])
    population_x = ((population - mean) / scale).float()
    clean_prediction = torch.cat([model(batch) for batch in population_x.split(1024)])
    empirical = centered_diagonal(
        lambda u, v: pair_model.pair_values(u, v, index), val_x[:, i].double(), val_x[:, j].double(),
    )
    return prediction, {
        "pair_mse": (weights @ pure_error.square() @ weights).item(),
        "signal_mse": (clean_prediction.double() - clean_y).square().mean().item(),
        "validation_rmse": rmse(model(val_x), val_y),
        "test_rmse": rmse(prediction, y),
        "pair_score": empirical.square().mean().sqrt().item(),
    }


def _fit(model, train, validation, init, rate, penalty):
    """Replay the frozen Adam recipe and continue to a fixed-update endpoint."""
    batch_seed = int(np.random.SeedSequence([c.BATCH_STREAM, init]).generate_state(1)[0])
    order_rng = torch.Generator().manual_seed(batch_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    x, y = train
    val_x, val_y = validation
    history = []
    best, stopped = float("inf"), None
    start = perf_counter()
    for epoch in range(1, c.MAX_EPOCHS + 1):
        model.train()
        for batch in torch.randperm(len(y), generator=order_rng).split(c.BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[batch]) - y[batch]).square().mean()
            if penalty:
                loss = loss + penalty * model.group_penalty()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            error = rmse(model(val_x), val_y)
        if not np.isfinite(error):
            raise RuntimeError(f"Nonfinite validation RMSE at epoch {epoch}")
        history.append(error)
        if stopped is None:
            if error < best:
                best, selected_epoch = error, epoch
                selected = deepcopy(model.state_dict())
            if epoch - selected_epoch >= c.PATIENCE:
                stopped = epoch
    return ({"selected": selected, "epoch_300": deepcopy(model.state_dict())},
            {"selected_epoch": selected_epoch, "stop_epoch": stopped or c.MAX_EPOCHS,
             "validation_history": history, "batch_seed": batch_seed, "training_seconds": perf_counter() - start})


def _provenance(follow, upstream, seed, init):
    key = f"{TASK}-{seed}-{init}-mgs_kan"
    meta = upstream["fits"][key]
    path = follow / meta["artifact"]
    assert sha256(path.read_bytes()).hexdigest() == meta["sha256"]
    return torch.load(path, weights_only=True, map_location="cpu")


def _verify(run, identity, splits, baseline):
    assert run["identity"] == identity
    kernels = identity["kernels"]
    model = _model(kernels, identity["init"]).eval()
    assert parameter_count(model) == run["parameters"] == 949 + 6 * (kernels - 2)
    train, validation, test = splits
    x, y = test
    assert torch.equal(run["test_target"], y)
    assert run["selection_cutoff"] == _threshold(train)
    training = run["training"]
    history = training["validation_history"]
    stop = training["stop_epoch"]
    best = min(range(stop), key=history.__getitem__) + 1
    assert len(history) == c.MAX_EPOCHS and best == training["selected_epoch"]
    assert stop == min(best + c.PATIENCE, c.MAX_EPOCHS)
    assert training["batch_seed"] == baseline["metrics"]["batch_seed"]
    for endpoint, epoch in zip(ENDPOINTS, (best, c.MAX_EPOCHS)):
        model.load_state_dict(run["checkpoint"][endpoint], strict=True)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        with torch.inference_mode():
            assert torch.equal(model(x), run["prediction"][endpoint])
            assert rmse(model(validation[0]), validation[1]) == history[epoch - 1]
        metrics = run["metrics"][endpoint]
        assert np.isfinite(list(metrics.values())).all()
        assert metrics["validation_rmse"] == history[epoch - 1]
        assert metrics["test_rmse"] == rmse(run["prediction"][endpoint], y)
    if (kernels, identity["lr"], identity["lambda"]) == BASELINE:
        assert history[:stop] == baseline["metrics"]["validation_history"]
        assert best == baseline["metrics"]["best_epoch"]
        assert torch.equal(run["prediction"]["selected"], baseline["test"]["prediction"])


def _job(args):
    output, follow, upstream, reference, protocol_id, seed, init, kernels, rate, penalty = args
    key = f"{seed}-{init}-k{kernels}-lr{rate:g}-reg{penalty:g}"
    identity = {"protocol_id": protocol_id, "seed": seed, "init": init, "kernels": kernels, "lr": rate, "lambda": penalty}
    path = output / "fits" / f"{key}.pt"
    splits, _ = make_data(TASK, seed, reference)
    train, validation, test = splits
    baseline = _provenance(follow, upstream, seed, init)
    if path.exists():
        run = torch.load(path, weights_only=True, map_location="cpu")
        _verify(run, identity, splits, baseline)
        return key, path, run["training"], run["metrics"], run["selection_cutoff"], True

    model = _model(kernels, init)
    checkpoint, training = _fit(model, train, validation, init, rate, penalty)
    feature_seed, _ = np.random.SeedSequence(seed).spawn(2)
    raw = np.random.default_rng(feature_seed).standard_normal((sum(c.SPLIT_SIZES), c.FEATURES))
    raw_train = raw[:c.SPLIT_SIZES[0]]
    mean, scale = map(torch.from_numpy, (raw_train.mean(axis=0), raw_train.std(axis=0)))
    assert torch.equal(((torch.from_numpy(raw_train) - mean) / scale).float(), train[0])
    prediction, results = {}, {}
    for endpoint, state in checkpoint.items():
        model.load_state_dict(state)
        prediction[endpoint], results[endpoint] = _evaluate(model, validation, test, mean, scale, reference)
    cutoff = _threshold(train)
    run = {"identity": identity, "parameters": parameter_count(model), "training": training,
           "checkpoint": checkpoint, "prediction": prediction, "test_target": test[1],
           "selection_cutoff": cutoff, "metrics": results}
    with _atomic_file(path) as temp:
        torch.save(run, temp)
        _verify(torch.load(temp, weights_only=True, map_location="cpu"), identity, splits, baseline)
    return key, path, training, results, cutoff, False


def run_ablation(root, follow, follow_protocol):
    """Run the prespecified 72 fits and return two endpoints per fit."""
    _setup_torch()
    _check_initialization()
    upstream_path = follow / "final" / "manifest.json"
    upstream = json.loads(upstream_path.read_text())
    assert upstream["protocol_id"] == follow_protocol["id"]
    baseline_tuning = json.loads((follow / "tuning" / "selection.json").read_text())["selected"][TASK]["mgs_kan"]
    assert baseline_tuning == {"lr": BASELINE[1], "regularization": BASELINE[2]}
    assert c.MAX_EPOCHS == 300 and c.VALIDATE_EVERY == 1 and c.DEVICE == "cpu"
    protocol = {"task": TASK, "pair": PAIR, "seeds": SEEDS, "inits": c.INIT_SEEDS,
                "kernels": KERNELS, "rates": RATES, "penalties": PENALTIES, "baseline": BASELINE,
                "extra_init_stream": EXTRA_STREAM, "population_seed": EVAL_SEED, "population_power": EVAL_POWER,
                "quadrature": QUADRATURE, "epochs": c.MAX_EPOCHS, "patience": c.PATIENCE, "batch_size": c.BATCH_SIZE,
                "runtime": _runtime(), "threads": c.THREADS, "workers": c.FOLLOWUP_WORKERS,
                "upstream_id": follow_protocol["id"],
                "upstream_manifest_sha256": sha256(upstream_path.read_bytes()).hexdigest(),
                "ablation_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    protocol = json.loads(json.dumps(protocol))
    protocol_id = _digest(protocol)
    output = Path(root) / protocol_id[:16]
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        _write_json(protocol_path, protocol)

    designs = list(product(SEEDS, c.INIT_SEEDS, KERNELS, RATES, PENALTIES))
    assert len(designs) == len(set(designs)) == 72
    jobs = [(output, follow, upstream, follow_protocol["reference"], protocol_id, *row) for row in designs]
    rows, manifest, reused = [], {}, 0
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=c.FOLLOWUP_WORKERS, mp_context=get_context("spawn"),
                             initializer=_setup_torch) as pool:
        for number, (design, result) in enumerate(zip(designs, pool.map(_job, jobs)), start=1):
            seed, init, kernels, rate, penalty = design
            key, path, training, endpoints, cutoff, was_reused = result
            reused += was_reused
            manifest[key] = {"artifact": str(path.relative_to(output)), "sha256": sha256(path.read_bytes()).hexdigest()}
            for endpoint, metrics in endpoints.items():
                rows.append({"fit": key, "seed": seed, "init": init, "kernels": kernels, "lr": rate, "lambda": penalty,
                             "endpoint": endpoint, "epoch": training["selected_epoch"] if endpoint == "selected" else c.MAX_EPOCHS,
                             "stop_epoch": training["stop_epoch"], "training_seconds": training["training_seconds"],
                             "selection_cutoff": cutoff, **metrics})
            if number % 9 == 0:
                print(f"{number}/{len(designs)} fits verified ({reused} reused)", flush=True)
    assert len(manifest) == 72 and len(rows) == 144
    assert {p.stem for p in (output / "fits").glob("*.pt")} == set(manifest)
    summary = {"protocol_id": protocol_id, "fits": manifest, "rows": rows}
    summary_path = output / "manifest.json"
    if not summary_path.exists() or json.loads(summary_path.read_text()) != summary:
        _write_json(summary_path, summary)
    return output, rows, perf_counter() - start
