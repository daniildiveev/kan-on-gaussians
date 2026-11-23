"""Matched interaction orders for Gaussian-state prediction."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
from itertools import combinations, product
import inspect
import json
import math
from multiprocessing import get_context
from pathlib import Path
import pickle
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch import nn
from torch.func import functional_call

from experiments import geometry_benchmark as benchmark, geometry_reference as ref
from experiments.hypergraph_order import GaussianBank
from src.experiment import _digest, _runtime, _setup_torch, _write_json

OUTPUT = Path("outputs/study/geometry_order")
MODELS = ("I", "II", "III", "Flat MLP")
COUNTS = dict(zip(MODELS, (2505, 2515, 2535, 2483), strict=True))
KERNELS = {"I": {10: 35}, "II": {1: 3, 2: 5}, "III": {1: 5, 2: 1, 3: 1}}
DATA_SEEDS, INIT_SEEDS = (9000, 10000, 11000), (11, 29, 47)
DEV_SEED, DEV_INITS = 2000, (11, 29)
RATES = (0.001, 0.003, 0.01)
WORKERS = 5
METRICS = ("w2", "mean_error", "cov_error")


class VectorBank(nn.Module):
    """A Gaussian basis with separate amplitudes for the five residual coordinates."""

    factor = GaussianBank.factor
    precision = GaussianBank.precision

    def __init__(self, order, kernels):
        super().__init__()
        groups = tuple(combinations(range(10), order))
        shape = (len(groups), kernels)
        self.register_buffer("indices", torch.tensor(groups))
        row, col = torch.tril_indices(order, order)
        self.register_buffer("factor_row", row)
        self.register_buffer("factor_col", col)
        if order == 1:
            center = torch.linspace(-0.75, 0.75, kernels).view(1, -1, 1)
            center = center + 0.025 * torch.randn(*shape, 1)
            raw = torch.zeros(*shape, 1)
        else:
            center = 0.55 * torch.randn(*shape, order)
            raw = 0.08 * torch.randn(*shape, len(row))
        self.center = nn.Parameter(center)
        self.raw = nn.Parameter(raw)
        self.amplitude = nn.Parameter(0.003 * torch.randn(*shape, 5))

    def forward(self, x):
        displacement = x[:, self.indices].unsqueeze(-2) - self.center
        projected = torch.einsum("ngki,gkij->ngkj", displacement, self.factor())
        kernels = torch.exp(-0.5 * projected.square().sum(dim=-1))
        return kernels.flatten(1) @ self.amplitude.flatten(0, 1)


class VectorResidual(nn.Module):
    def __init__(self, kernels):
        super().__init__()
        self.affine = nn.Linear(10, 5)
        nn.init.normal_(self.affine.weight, std=0.01)
        nn.init.zeros_(self.affine.bias)
        self.banks = nn.ModuleList(VectorBank(order, count) for order, count in kernels.items())

    def forward(self, x):
        residual = self.affine(x)
        for bank in self.banks:
            residual = residual + bank(x)
        return residual


class GeometryOrder(nn.Module):
    forward = benchmark.GeometryMGSKAN.forward

    def __init__(self, kernels):
        super().__init__()
        self.register_buffer("coord_scale", torch.tensor([1.5, 1.5, 0.7, 0.7, 0.7]))
        self.mgs_residual = VectorResidual(kernels)


def make_model(name, seed):
    with torch.random.fork_rng(devices=[]), ref.float64():
        torch.manual_seed(seed)
        if name == "Flat MLP":
            return ref.FlatMLP()
        if name == "II":
            return benchmark.GeometryMGSKAN()
        return GeometryOrder(KERNELS[name])


def freeze_protocol(root=OUTPUT):
    schedule = {key: ref.STAGES["lr_search"][key] for key in ref.RUN_FIELDS}
    schedule.update(clip_norm=5.0, adam_betas=(0.9, 0.999), adam_eps=1e-8)
    protocol = {
        "tasks": ref.TASKS, "models": MODELS, "parameters": COUNTS,
        "kernels_by_order": KERNELS, "input_coordinates": 10, "residual_coordinates": 5,
        "data_seeds": DATA_SEEDS, "init_seeds": INIT_SEEDS,
        "development_seed": DEV_SEED, "development_initializations": DEV_INITS, "learning_rates": RATES,
        "schedule": schedule, "regularization": 0,
        "checkpoint_rule": "earliest minimum validation W2 squared, including step 0",
        "rate_rule": "smallest mean best validation W2 squared, then smallest learning rate",
        "batch_rng": "torch.Generator seeded with train_seed + 10000, draws with replacement",
        "runtime": {**_runtime(), "device": "cpu", "threads": 1, "workers": WORKERS},
        "code_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), Path(ref.__file__), Path(benchmark.__file__))},
        "shared_factor_sha256": sha256("".join(inspect.getsource(f) for f in
             (GaussianBank.factor, GaussianBank.precision)).encode()).hexdigest(),
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


def _fit(model, cfg, data):
    train, validation = data["train"], data["val"]
    x, y = train
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.adam_betas, eps=cfg.adam_eps)
    rng = torch.Generator().manual_seed(cfg.train_seed + 10000)
    order = ref.batch_indices(cfg.n_train, cfg.batch, cfg.steps, rng)
    best, history = math.inf, []
    start = perf_counter()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            row = {"step": step, **benchmark._curve_row(model, train, validation)}
            history.append(row)
            if row["val_w2"] < best:
                best, selected = row["val_w2"], step
                checkpoint = deepcopy(model.state_dict())
        if step == cfg.steps:
            break
        idx = order[step]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*(v[idx] for v in x))
        ref.w2sq(prediction, tuple(v[idx] for v in y)).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm, error_if_nonfinite=True)
        optimizer.step()
    model.load_state_dict(checkpoint, strict=True)
    return model, selected, history, perf_counter() - start


def _identity(protocol, stage, cfg):
    return {"protocol_id": protocol["id"], "stage": stage, "key": benchmark._run_key(cfg),
            "config": json.loads(json.dumps(vars(cfg)))}


@torch.inference_mode()
def _save_fit(path, identity, data, fit):
    model, step, history, seconds = fit
    splits, metrics = {}, {}
    names = ("train", "val", "test") if identity["stage"] == "final" else ("train", "val")
    for name in names:
        inputs, target = data[name]
        prediction = model(*inputs)
        splits[name] = {"prediction": prediction, "target": target}
        metrics[name] = benchmark._metrics(prediction, target)
    benchmark._save_pt(path, {"identity": identity, "checkpoint": model.state_dict(),
                              "selected_step": step, "history": history, "splits": splits, "metrics": metrics,
                              "training_seconds": seconds})


@torch.inference_mode()
def _verify(path, identity, cfg, data):
    result = torch.load(path, weights_only=True, map_location="cpu")
    assert result["identity"] == identity
    steps = sorted(set(range(0, cfg.steps + 1, cfg.eval_every)) | {cfg.steps})
    history = result["history"]
    assert [row["step"] for row in history] == steps
    assert all(math.isfinite(r[k]) for r in history for k in ref.CURVE_COLS)
    best = min(history, key=lambda row: (row["val_w2"], row["step"]))
    assert best["step"] == result["selected_step"]
    model = make_model(cfg.model, cfg.train_seed)
    model.load_state_dict(result["checkpoint"], strict=True)
    assert ref.count_params(model) == COUNTS[cfg.model]
    assert all(p.dtype == torch.float64 and torch.isfinite(p).all() for p in model.parameters())
    names = {"train", "val", "test"} if identity["stage"] == "final" else {"train", "val"}
    assert set(result["splits"]) == set(result["metrics"]) == names
    for name in names:
        inputs, target = data[name]
        previous = result["splits"][name]
        assert all(torch.equal(a, b) for a, b in zip(target, previous["target"], strict=True))
        for a, b in zip(model(*inputs), previous["prediction"], strict=True):
            torch.testing.assert_close(a, b, atol=benchmark.ATOL, rtol=benchmark.RTOL)
        for key, value in benchmark._metrics(previous["prediction"], target).items():
            assert math.isclose(value, result["metrics"][name][key],
                                abs_tol=benchmark.ATOL, rel_tol=benchmark.RTOL)
    assert math.isclose(best["val_w2"], result["metrics"]["val"]["w2"],
                        abs_tol=benchmark.ATOL, rel_tol=benchmark.RTOL)
    assert math.isfinite(result["training_seconds"]) and result["training_seconds"] >= 0
    return result


def _configs(protocol, stage, selected):
    for model, task in product(MODELS, ref.TASKS):
        if stage == "development":
            seeds, inits, rates = (DEV_SEED,), DEV_INITS, RATES
        else:
            seeds, inits, rates = DATA_SEEDS, INIT_SEEDS, (selected[task][model],)
        for seed, init, lr in product(seeds, inits, rates):
            yield ref.RunCfg(model, task, seed, init, lr, **protocol["schedule"])


def _job(output, protocol, stage, cfg):
    with ref.float64():
        identity = _identity(protocol, stage, cfg)
        path = output / stage / "fits" / f"{identity['key']}.pt"
        data = ref.make_dataset(cfg.task, cfg.data_seed, cfg.n_train, cfg.n_val, cfg.n_test)
        try:
            saved = _verify(path, identity, cfg, data)
            reused = True
        except (OSError, EOFError, AssertionError, IndexError, KeyError, pickle.UnpicklingError,
                RuntimeError, TypeError, ValueError):
            _save_fit(path, identity, data, _fit(make_model(cfg.model, cfg.train_seed), cfg, data))
            saved = _verify(path, identity, cfg, data)
            reused = False
        return {"key": identity["key"], "task": cfg.task, "model": cfg.model,
                "data_seed": cfg.data_seed, "train_seed": cfg.train_seed, "lr": cfg.lr,
                "parameters": COUNTS[cfg.model], "selected_step": saved["selected_step"],
                "metrics": saved["metrics"], "training_seconds": saved["training_seconds"],
                "device": protocol["runtime"]["cpu"], "provenance": "geometry order " + stage,
                "artifact": str(path.relative_to(output)), "sha256": sha256(path.read_bytes()).hexdigest()}, reused


def _run(output, protocol, stage, selected=None):
    assert json.loads((output / "protocol.json").read_text()) == protocol
    configs = list(_configs(protocol, stage, selected))
    rows, reused = {}, 0
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn"),
                             initializer=_setup_torch) as pool:
        jobs = [pool.submit(_job, output, protocol, stage, cfg) for cfg in configs]
        for job in as_completed(jobs):
            row, cached = job.result()
            assert row["key"] not in rows
            rows[row["key"]] = row
            reused += cached
            if len(rows) % 12 == 0:
                print(f"{stage}: {len(rows)}/{len(configs)} verified ({reused} reused)", flush=True)
    assert set(rows) == {benchmark._run_key(cfg) for cfg in configs}
    assert {p.stem for p in (output / stage / "fits").glob("*.pt")} == set(rows)
    manifest = {"protocol_id": protocol["id"], "fits": [rows[k] for k in sorted(rows)]}
    _write_json(output / stage / "manifest.json", manifest)
    print(f"{stage}: {len(configs)} fits in {perf_counter() - start:.1f} s on CPU", flush=True)
    return manifest["fits"]


def tune(output, protocol):
    rows = _run(output, protocol, "development")
    assert len(rows) == 72
    selection = {task: {} for task in ref.TASKS}
    trials = []
    for task, model in product(ref.TASKS, MODELS):
        scores = {}
        for lr in RATES:
            matching = [r for r in rows if (r["task"], r["model"], r["lr"]) == (task, model, lr)]
            assert {r["train_seed"] for r in matching} == set(DEV_INITS) and len(matching) == 2
            scores[lr] = mean(r["metrics"]["val"]["w2"] for r in matching)
            trials.append({"task": task, "model": model, "lr": lr, "mean_val_w2": scores[lr]})
        selection[task][model] = min(RATES, key=lambda lr: (scores[lr], lr))
    choice = {"protocol_id": protocol["id"], "selected": selection, "trials": trials}
    path = output / "development" / "selection.json"
    if path.exists():
        assert json.loads(path.read_text()) == choice
    else:
        _write_json(path, choice)
    return choice


def run_final(output, protocol, choice):
    assert choice["protocol_id"] == protocol["id"]
    assert json.loads((output / "development" / "selection.json").read_text()) == choice
    rows = _run(output, protocol, "final", choice["selected"])
    assert len(rows) == 108
    benchmark._fit_csv(output, rows)
    return rows


def run_checks(output, protocol):
    _setup_torch()
    counts, gradient_checks, smoke = {}, {}, []
    with ref.float64(), TemporaryDirectory() as temp:
        for name in MODELS:
            model = make_model(name, INIT_SEEDS[0])
            counts[name] = ref.count_params(model)
            mu, cov = ref.random_gaussians(2, seed=27502, device=torch.device("cpu"))
            names, params = zip(*model.named_parameters())
            assert torch.autograd.gradcheck(
                lambda m, c, *p: functional_call(model, dict(zip(names, p, strict=True)), (m, c)),
                (mu.requires_grad_(), cov.requires_grad_(), *params), fast_mode=True,
            )
            gradient_checks[name] = True
            if name in ("I", "III"):
                for bank, (arity, kernels) in zip(model.mgs_residual.banks, KERNELS[name].items(), strict=True):
                    assert bank.indices.tolist() == [list(g) for g in combinations(range(10), arity)]
                    P = bank.precision().detach()
                    assert P.shape == (math.comb(10, arity), kernels, arity, arity)
                    assert (torch.linalg.eigvalsh(P) > 0).all()
                    z = torch.tanh(ref.encode_gaussian(mu, cov) / model.coord_scale).flatten(1)
                    d = z[:, bank.indices].unsqueeze(-2) - bank.center
                    expected = torch.exp(-0.5 * torch.einsum("ngki,gkij,ngkj->ngk", d, P, d))
                    torch.testing.assert_close(bank(z), expected.flatten(1) @ bank.amplitude.flatten(0, 1),
                                               atol=1e-13, rtol=1e-13)
            with torch.no_grad():
                residual = model.net[-1] if name == "Flat MLP" else model.mgs_residual
                for parameter in residual.parameters():
                    parameter.zero_()
                pred_mu, pred_cov = model(mu, cov)
                torch.testing.assert_close(pred_mu, mu.mean(1, keepdim=True), atol=1e-13, rtol=1e-13)
                torch.testing.assert_close(pred_cov, cov.mean(1, keepdim=True), atol=1e-13, rtol=1e-13)
                assert (torch.linalg.eigvalsh(pred_cov) > 0).all()
        assert counts == COUNTS and (max(counts.values()) - min(counts.values())) / COUNTS["II"] < .05
        for task, model in product(ref.TASKS, MODELS):
            schedule = {**protocol["schedule"], "steps": 20}
            cfg = ref.RunCfg(model, task, DEV_SEED, DEV_INITS[0], 0.003, **schedule)
            data = ref.make_dataset(task, cfg.data_seed, cfg.n_train, cfg.n_val, cfg.n_test)
            identity = _identity(protocol, "smoke", cfg)
            path = Path(temp) / "smoke.pt"
            _save_fit(path, identity, data, _fit(make_model(model, cfg.train_seed), cfg, data))
            result = _verify(path, identity, cfg, data)
            smoke.append({"task": task, "model": model, "val_w2": result["metrics"]["val"]["w2"]})
    checks = {"parameters": counts, "gradcheck": gradient_checks, "smoke": smoke}
    _write_json(output / "checks.json", checks)
    return checks


def report(rows, choice):
    from IPython.display import Markdown, display
    import matplotlib.pyplot as plt

    assert len(rows) == 108
    table = benchmark._table
    display(Markdown("### Зафиксированные скорости обучения и бюджет"))
    table(("Модель", "Параметры", *ref.TASKS),
          [(m, COUNTS[m], *(choice["selected"][t][m] for t in ref.TASKS)) for m in MODELS])

    averages = {(t, m, s): benchmark._aggregates(rows, t, m, seed=s)
                for t, m, s in product(ref.TASKS, MODELS, DATA_SEEDS)}
    totals = {(t, m): benchmark._aggregates(rows, t, m) for t, m in product(ref.TASKS, MODELS)}
    display(Markdown("### Тестовая ошибка гауссовского состояния"))
    table(("Задача", "Модель", "W₂²", "Центр", "Ковариация"),
          [(t, m, *(f"{value:.7f}" for value in totals[t, m])) for t, m in totals])

    references = ("II", "I", "Flat MLP")
    differences = {(t, s, m): [a - b for a, b in zip(averages[t, "III", s], averages[t, m, s], strict=True)]
                   for t, s, m in product(ref.TASKS, DATA_SEEDS, references)}
    display(Markdown("### Парные разности III минус модель сравнения"))
    table(("Задача", "Seed", "Модель сравнения", "Δ W₂²", "Δ центр", "Δ ковариация"),
          [(*key, *(f"{value:+.7f}" for value in values)) for key, values in differences.items()])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), layout="constrained")
    for axis, task in zip(axes, ref.TASKS, strict=True):
        axis.axhline(0, color=".45", linewidth=.8)
        for offset, seed in zip((-.2, 0, .2), DATA_SEEDS, strict=True):
            axis.scatter([i + offset for i in range(3)],
                         [differences[task, seed, m][0] for m in references], label=str(seed))
        axis.set(xticks=range(3), xticklabels=["III−II", "III−I", "III−MLP"], title=task)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Разность тестовой W₂²")
    axes[-1].legend(title="Seed")
    plt.show()
    plt.close(fig)

    display(Markdown("### Вывод по прогнозу"))
    for t in ref.TASKS:
        lead = min(MODELS, key=lambda m: totals[t, m][0])
        delta = [a - b for a, b in zip(totals[t, "III"], totals[t, "II"], strict=True)]
        wins = sum(differences[t, s, "II"][0] < 0 for s in DATA_SEEDS)
        display(Markdown(f"- **{t}**. Лучшая модель по среднему W₂² — {lead}. "
                         f"У III ниже W₂², чем у II, на {wins}/3 выборках. "
                         f"Средняя разность {delta[0]:+.7f}, из неё центр {delta[1]:+.7f}, "
                         f"ковариация {delta[2]:+.7f}."))
