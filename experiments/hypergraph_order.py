"""Matched scalar pilot for single-layer MGS-KAN interaction order."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, product
import inspect
import json
from math import sqrt
from multiprocessing import get_context
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from src.experiment import _atomic_file, _digest, _runtime, _setup_torch, _write_json, rmse
from src.models import MGS_KAN, parameter_count

OUTPUT = Path("outputs/study/hypergraph_order")
TASKS = ("additive", "pair", "triple")
DATA_SEEDS, INIT_SEEDS = (61011, 61013, 61017), (101, 103, 107)
SIZES = (1024, 256, 1024)
LR, BATCH, EPOCHS, PATIENCE = 0.003, 64, 120, 18
NOISE_SD, QUADRATURE_NODES = 0.1, 80
BATCH_STREAM, SCORE_SEED, SCORE_ROWS = 19267, 27501, 1024
WORKERS = 5
COUNTS = {"I": 515, "II": 521, "III": 517, "MLP": 516}
KERNELS = {"I": {4: 34}, "II": {1: 4, 2: 13}, "III": {1: 4, 2: 4, 3: 8}}
TRIPLES = tuple(combinations(range(4), 3))
PLANTED = (0, 1, 3)


class PairBank(nn.Module):
    """Keep the established pair kernel and factor convention."""

    factor_entries = MGS_KAN.factor_entries
    precision = MGS_KAN.precision
    pair_values = MGS_KAN.pair_values

    def __init__(self, groups, kernels):
        super().__init__()
        shape = (len(groups), kernels)
        self.register_buffer("indices", torch.tensor(groups))
        self.pair_center = nn.Parameter(0.55 * torch.randn(*shape, 2, dtype=torch.float64))
        self.pair_amplitude = nn.Parameter(0.035 * torch.randn(shape, dtype=torch.float64))
        self.pair_precision_raw = nn.Parameter(0.08 * torch.randn(*shape, 3, dtype=torch.float64))

    def forward(self, coordinates):
        u, v = coordinates.unbind(dim=-1)
        return self.pair_values(u, v)


class GaussianBank(nn.Module):
    """Groups of one arity with a shared kernel count and full precision per kernel."""

    def __init__(self, groups, kernels):
        super().__init__()
        width = len(groups[0])
        shape = (len(groups), kernels)
        self.register_buffer("indices", torch.tensor(groups))
        row, column = torch.tril_indices(width, width)
        self.register_buffer("factor_row", row)
        self.register_buffer("factor_col", column)
        if width == 1:
            center = torch.linspace(-1.5, 1.5, kernels, dtype=torch.float64).view(1, -1, 1)
            center = center.repeat(len(groups), 1, 1)
            raw = torch.zeros(*shape, 1, dtype=torch.float64)
        else:
            center = 0.55 * torch.randn(*shape, width, dtype=torch.float64)
            raw = 0.08 * torch.randn(*shape, len(row), dtype=torch.float64)
        self.center = nn.Parameter(center)
        self.raw = nn.Parameter(raw)
        self.amplitude = nn.Parameter((0.09 if width == 1 else 0.035)
                                      * torch.randn(shape, dtype=torch.float64))

    def factor(self):
        width = self.indices.shape[1]
        factor = self.raw.new_zeros((*self.raw.shape[:-1], width, width))
        row, column = self.factor_row, self.factor_col
        factor[..., row, column] = torch.where(row == column, self.raw.exp(), self.raw)
        return factor

    def precision(self):
        factor = self.factor()
        return factor @ factor.mT

    def forward(self, coordinates):
        displacement = coordinates.unsqueeze(-2) - self.center
        projected = torch.einsum("...gki,gkij->...gkj", displacement, self.factor())
        kernel = torch.exp(-0.5 * projected.square().sum(dim=-1))
        return (kernel * self.amplitude).sum(dim=-1)


class OrderLayer(nn.Module):
    def __init__(self, kernels):
        super().__init__()
        self.affine = nn.Linear(4, 1, dtype=torch.float64)
        self.banks = nn.ModuleList(
            (PairBank if order == 2 else GaussianBank)(tuple(combinations(range(4), order)), count)
            for order, count in kernels.items()
        )

    def forward(self, x):
        result = self.affine(x).squeeze(-1)
        for bank in self.banks:
            result = result + bank(x[..., bank.indices]).sum(dim=-1)
        return result


def make_model(name, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if name in KERNELS:
            return OrderLayer(KERNELS[name])
        assert name == "MLP"
        return nn.Sequential(nn.Linear(4, 19, dtype=torch.float64), nn.ReLU(),
                             nn.Linear(19, 20, dtype=torch.float64), nn.ReLU(),
                             nn.Linear(20, 1, dtype=torch.float64), nn.Flatten(start_dim=0))


@lru_cache(maxsize=1)
def _normalizer():
    nodes, weights = np.polynomial.hermite.hermgauss(QUADRATURE_NODES)
    return sqrt(float(weights @ np.tanh(sqrt(2) * nodes)**2 / sqrt(np.pi)))


def _splits(task, seed, normalizer):
    x_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    x = np.random.default_rng(x_seed).standard_normal((sum(SIZES), 4))
    noise = np.random.default_rng(noise_seed).normal(0, NOISE_SD, len(x))
    u0, u1, _, u3 = (np.tanh(x) / normalizer).T
    signal = {"additive": (u0 + u1 + u3) / sqrt(3),
              "pair": u0 * u1, "triple": u0 * u1 * u3}[task]
    return tuple(zip(torch.from_numpy(x).split(SIZES), torch.from_numpy(signal + noise).split(SIZES)))


def freeze_protocol(root=OUTPUT):
    protocol = {
        "tasks": TASKS, "data_seeds": DATA_SEEDS, "init_seeds": INIT_SEEDS, "split_sizes": SIZES,
        "groups": {name: [[list(group), count] for size, count in orders.items()
                         for group in combinations(range(4), size)] for name, orders in KERNELS.items()},
        "mlp_widths": [4, 19, 20, 1], "parameters": COUNTS,
        "training": {"dtype": "float64", "optimizer": "Adam", "lr": LR, "batch_size": BATCH,
                     "max_epochs": EPOCHS, "patience": PATIENCE, "weight_decay": 0, "group_penalty": 0,
                     "validate_every": 1, "checkpoint": "earliest minimum validation MSE",
                     "batch_stream": BATCH_STREAM},
        "generator": {"quadrature_nodes": QUADRATURE_NODES, "normalizer": _normalizer(),
                      "noise_sd": NOISE_SD, "input_scaling": "none"},
        "triple_score": {"seed": SCORE_SEED, "rows": SCORE_ROWS},
        "runtime": {**_runtime(), "device": "cpu", "workers": WORKERS, "torch_threads": 1},
        "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_code_sha256": sha256("".join(inspect.getsource(f) for f in (
            MGS_KAN.pair_values, MGS_KAN.factor_entries, MGS_KAN.precision,
            _atomic_file, _setup_torch, parameter_count, rmse,
        )).encode()).hexdigest(),
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


def _mixed_difference(function, x, alternate):
    terms = []
    for mask in product((False, True), repeat=3):
        coordinates = torch.where(torch.tensor(mask), alternate, x)
        terms.append((-1)**sum(mask) * function(coordinates))
    return torch.stack(terms).sum(dim=0)


@lru_cache(maxsize=1)
def _score_inputs():
    rng = np.random.default_rng(SCORE_SEED)
    return tuple(torch.from_numpy(rng.standard_normal((SCORE_ROWS, 4))) for _ in range(2))


@torch.inference_mode()
def _scores(model):
    bank = model.banks[-1]
    x, alternate = (x[:, bank.indices] for x in _score_inputs())
    scores = (_mixed_difference(bank, x, alternate).square().mean(dim=0) / 8).sqrt().tolist()
    ranked = sorted(zip(TRIPLES, scores, strict=True), key=lambda row: (-row[1], row[0]))
    return [{"group": list(group), "score": value, "rank": rank}
            for rank, (group, value) in enumerate(ranked, start=1)]


def _batch_seed(seed, init):
    return int(np.random.SeedSequence([BATCH_STREAM, seed, init]).generate_state(1)[0])


def _mse(prediction, target):
    return (prediction - target).square().mean()


def _fit(identity, splits, epochs):
    seed, init, name = (identity[k] for k in ("seed", "init", "model"))
    model = make_model(name, init)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    batch_seed = _batch_seed(seed, init)
    rng = torch.Generator().manual_seed(batch_seed)
    (x, y), (val_x, val_y), (test_x, target) = splits
    best, history = float("inf"), []
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        for indices in torch.randperm(len(x), generator=rng).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            _mse(model(x[indices]), y[indices]).backward()
            optimizer.step()
        with torch.no_grad():
            score = _mse(model(val_x), val_y).item()
        assert np.isfinite(score), (identity, epoch, score)
        history.append(score)
        if score < best:
            best, selected = score, epoch
            checkpoint = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch - selected == PATIENCE:
            break
    elapsed = perf_counter() - started
    model.load_state_dict(checkpoint, strict=True)
    with torch.inference_mode():
        prediction = model(test_x)
    return {"identity": identity, "checkpoint": checkpoint, "batch_seed": batch_seed,
            "best_epoch": selected, "epochs_run": epoch, "validation_mse": best,
            "validation_history": history, "training_seconds": elapsed,
            "rows": torch.arange(sum(SIZES[:2]), sum(SIZES)), "target": target, "prediction": prediction,
            "test_rmse": rmse(prediction, target), "parameters": parameter_count(model),
            "triple_scores": _scores(model) if name == "III" else []}


@torch.inference_mode()
def _verify(path, identity, splits, epochs=EPOCHS):
    fit = torch.load(path, map_location="cpu", weights_only=True)
    assert fit["identity"] == identity
    name = identity["model"]
    model = make_model(name, identity["init"]).eval()
    model.load_state_dict(fit["checkpoint"], strict=True)
    assert fit["parameters"] == parameter_count(model) == COUNTS[name]
    assert all(p.dtype == torch.float64 and torch.isfinite(p).all() for p in model.parameters())
    if name in KERNELS:
        for bank, (order, kernels) in zip(model.banks, KERNELS[name].items(), strict=True):
            assert bank.indices.tolist() == [list(g) for g in combinations(range(4), order)]
            precision = bank.precision()
            assert precision.shape == (len(bank.indices), kernels, order, order)
            assert (torch.linalg.eigvalsh(precision) > 0).all()
    stop, best, history = (fit[k] for k in ("epochs_run", "best_epoch", "validation_history"))
    assert 0 < stop == len(history) <= epochs and np.isfinite(history).all()
    assert best == min(range(stop), key=history.__getitem__) + 1
    assert stop == min(epochs, best + PATIENCE)
    assert np.isfinite(fit["training_seconds"]) and fit["training_seconds"] >= 0
    assert fit["batch_seed"] == _batch_seed(identity["seed"], identity["init"])
    _, (val_x, val_y), (test_x, target) = splits
    assert _mse(model(val_x), val_y).item() == fit["validation_mse"] == history[best - 1]
    assert torch.equal(fit["rows"], torch.arange(sum(SIZES[:2]), sum(SIZES)))
    assert torch.equal(fit["target"], target) and torch.equal(fit["prediction"], model(test_x))
    assert np.isfinite(fit["test_rmse"]) and rmse(fit["prediction"], target) == fit["test_rmse"]
    scores = _scores(model) if name == "III" else []
    assert fit["triple_scores"] == scores and all(np.isfinite(r["score"]) for r in scores)
    return fit


def _write_fit(path, identity, splits, epochs=EPOCHS):
    fit = _fit(identity, splits, epochs)
    with _atomic_file(path) as temp:
        torch.save(fit, temp)
        _verify(temp, identity, splits, epochs)
    return fit


def _identity(protocol, task, seed, init, model):
    return {"protocol_id": protocol["id"], "task": task, "seed": seed, "init": init, "model": model}


def run_checks(output, protocol):
    _setup_torch()
    x = torch.from_numpy(np.random.default_rng(27500).standard_normal((3, 4))).requires_grad_()
    counts, min_eigenvalue, explicit_error = {}, {}, {}
    for name in COUNTS:
        model = make_model(name, INIT_SEEDS[0])
        counts[name] = parameter_count(model)
        parameter_names, parameters = zip(*model.named_parameters())
        assert torch.autograd.gradcheck(
            lambda inputs, *p: functional_call(model, dict(zip(parameter_names, p, strict=True)), inputs),
            (x, *parameters),
        )
        if name == "MLP":
            continue
        assert [bank.indices.tolist() for bank in model.banks] == [
            [list(group) for group in combinations(range(4), size)] for size in KERNELS[name]
        ]
        with torch.no_grad():
            errors, minima = [], []
            for bank in model.banks:
                P = bank.precision()
                torch.testing.assert_close(P, P.mT, atol=1e-15, rtol=0)
                minima.append(torch.linalg.eigvalsh(P).min().item())
                z = x[:, bank.indices]
                center = bank.pair_center if isinstance(bank, PairBank) else bank.center
                amplitude = bank.pair_amplitude if isinstance(bank, PairBank) else bank.amplitude
                d = z.unsqueeze(-2) - center
                q = torch.einsum("ngki,gkij,ngkj->ngk", d, P, d)
                expected = (amplitude * torch.exp(-0.5 * q)).sum(-1)
                errors.append((bank(z) - expected).abs().max().item())
            assert min(minima) > 0 and max(errors) < 1e-14
            min_eigenvalue[name], explicit_error[name] = min(minima), max(errors)
    assert counts == COUNTS
    x, alternate = (x[:, :3] for x in _score_inputs())
    lower = _mixed_difference(lambda p: p[..., 0] * p[..., 1], x, alternate)
    pure = _mixed_difference(lambda p: p.prod(dim=-1), x, alternate)
    lower_score = (lower.square().mean().item() / 8)**0.5
    assert lower_score < 1e-14
    torch.testing.assert_close(pure, (x - alternate).prod(dim=-1), rtol=1e-12, atol=1e-14)
    smoke = []
    with TemporaryDirectory() as temp:
        for task, name in product(TASKS, COUNTS):
            identity = _identity(protocol, task, DATA_SEEDS[0], INIT_SEEDS[0], name)
            splits = _splits(task, identity["seed"], protocol["generator"]["normalizer"])
            fit = _write_fit(Path(temp) / "smoke.pt", identity, splits, epochs=2)
            smoke.append({"task": task, "model": name, "epochs": fit["epochs_run"],
                          "validation_mse": fit["validation_mse"]})
    checks = {"parameters": counts, "minimum_precision_eigenvalues": min_eigenvalue,
              "quadratic_form_max_errors": explicit_error, "gradcheck": "all four models passed",
              "lower_order_score": lower_score, "smoke_runs": smoke,
              "checkpoint_reload": "all smoke fits passed"}
    _write_json(output / "checks.json", checks)
    return checks


def _job(output, protocol, task, seed, init, name):
    key = f"{task}-{seed}-{init}-{name}"
    path = output / "fits" / f"{key}.pt"
    identity = _identity(protocol, task, seed, init, name)
    splits = _splits(task, seed, protocol["generator"]["normalizer"])
    try:
        fit = _verify(path, identity, splits)
        reused = True
    except (OSError, EOFError, AssertionError, IndexError, KeyError, pickle.UnpicklingError,
            RuntimeError, TypeError, ValueError) as error:
        if path.exists():
            print(f"Rebuilding {key}: failed verification ({type(error).__name__}).", flush=True)
        fit = _write_fit(path, identity, splits)
        reused = False
    metrics = {k: fit[k] for k in ("best_epoch", "epochs_run", "validation_mse", "test_rmse",
                                  "parameters", "training_seconds", "batch_seed", "triple_scores")}
    return key, {**identity, **metrics, "artifact": str(path.relative_to(output)),
                 "sha256": sha256(path.read_bytes()).hexdigest()}, reused


def run_final(output, protocol):
    _setup_torch()
    assert json.loads((output / "protocol.json").read_text()) == protocol
    jobs = list(product(TASKS, DATA_SEEDS, INIT_SEEDS, COUNTS))
    fits, reused = {}, 0
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn"),
                             initializer=_setup_torch) as pool:
        futures = [pool.submit(_job, output, protocol, *job) for job in jobs]
        for future in as_completed(futures):
            key, row, was_reused = future.result()
            assert key not in fits
            fits[key] = row
            reused += was_reused
            if len(fits) % 12 == 0:
                print(f"{len(fits)}/{len(jobs)} fits verified ({reused} reused)", flush=True)
    assert len(fits) == 108 and set(fits) == {"-".join(map(str, job)) for job in jobs}
    assert {p.stem for p in (output / "fits").glob("*.pt")} == set(fits)
    manifest = {"protocol_id": protocol["id"], "fits": dict(sorted(fits.items()))}
    _write_json(output / "manifest.json", manifest)
    print(f"Completed in {perf_counter() - started:.1f} s on CPU.", flush=True)
    return manifest


def report(manifest):
    from statistics import mean

    from IPython.display import Markdown, display
    import matplotlib.pyplot as plt

    def table(title, headings, rows):
        lines = [f"### {title}", "", "| " + " | ".join(headings) + " |",
                 "| " + " | ".join("---" for _ in headings) + " |"]
        lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
        display(Markdown("\n".join(lines)))

    fits = list(manifest["fits"].values())
    assert len(fits) == 108
    parameter_rows = []
    for model in COUNTS:
        counts = {f["parameters"] for f in fits if f["model"] == model}
        assert counts == {COUNTS[model]}
        parameter_rows.append((model, next(iter(counts))))
    table("Обучаемые параметры", ("Модель", "Параметры"), parameter_rows)

    averages = {(task, seed, model): mean(f["test_rmse"] for f in fits
                if (f["task"], f["seed"], f["model"]) == (task, seed, model))
                for task, seed, model in product(TASKS, DATA_SEEDS, COUNTS)}
    table("Прогноз на тестовой выборке", ("Задача", *COUNTS),
          [(t, *(f"{mean(averages[t, s, m] for s in DATA_SEEDS):.5f}" for m in COUNTS)) for t in TASKS])
    comparators = ("II", "I", "MLP")
    deltas = {(t, s, m): averages[t, s, "III"] - averages[t, s, m]
              for t, s, m in product(TASKS, DATA_SEEDS, comparators)}
    table("Парные разности RMSE. Меньше нуля в пользу III", ("Задача", "Seed", "III−II", "III−I", "III−MLP"),
          [(t, s, *(f"{deltas[t, s, m]:+.5f}" for m in comparators)) for t, s in product(TASKS, DATA_SEEDS)])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    for axis, task in zip(axes, TASKS, strict=True):
        axis.axhline(0, color=".5", linewidth=0.9)
        for offset, seed in zip((-.16, 0., .16), DATA_SEEDS, strict=True):
            axis.scatter(np.arange(3) + offset, [deltas[task, seed, m] for m in comparators], label=str(seed))
        axis.set(xticks=range(3), xticklabels=["III−II", "III−I", "III−MLP"], title=task)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Разность тестовой RMSE")
    axes[-1].legend(title="Seed")
    plt.show()
    plt.close(fig)

    triples = sorted((f for f in fits if f["model"] == "III"), key=lambda f: (f["seed"], f["init"]))
    target = [(f, next(r for r in f["triple_scores"] if tuple(r["group"]) == PLANTED))
              for f in triples if f["task"] == "triple"]
    table("Восстановление тройки (0,1,3)", ("Seed данных", "Инициализация", "Ранг из 4", "Сила"),
          [(f["seed"], f["init"], r["rank"], f"{r['score']:.5f}") for f, r in target])
    controls = {(f["task"], f["seed"], f["init"]): f["triple_scores"][0]["score"] for f in triples}
    table("Максимальная сила тройного вклада на контролях", ("Seed данных", "Инициализация", "additive", "pair"),
          [(s, i, *(f"{controls[t, s, i]:.5f}" for t in TASKS[:2])) for s, i in product(DATA_SEEDS, INIT_SEEDS)])

    verdict = ["### Вывод по прогнозу", ""]
    for task in TASKS:
        outcomes = []
        for model in comparators:
            values = [deltas[task, seed, model] for seed in DATA_SEEDS]
            outcomes.append(f"III−{model} {mean(values):+.5f}, побед III {sum(v < 0 for v in values)}/3")
        verdict.append(f"- **{task}**. " + ", ".join(outcomes) + ".")
    rank1 = sum(r["rank"] == 1 for _, r in target)
    verdict += ["", "### Вывод по восстановлению", "",
                f"Тройка (0,1,3) заняла первое место в {rank1}/9 запусков на triple.",
                "Для контролей приведены величины вкладов без порога отбора.",
                "Три независимые выборки дают описательное сравнение."]
    display(Markdown("\n".join(verdict)))
