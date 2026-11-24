"""Geometry with a known interaction in the encoded residual."""

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from math import sqrt
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from experiments import geometry_benchmark as base, geometry_order as previous, geometry_reference as ref
from experiments.hypergraph_order import _mixed_difference, _normalizer
from src.experiment import _digest, _setup_torch, _write_json

OUTPUT = Path("outputs/study/planted_geometry")
TASKS = ("center_triple", "covariance_triple", "pair_control")
MODELS = ("II", "III", "Flat MLP")
DATA_SEEDS = (12001, 12003, 12007, 12011, 12013)
DEV_SEED = 11991
SIZES = (768, 256, 1024)
TRIPLE = (0, 2, 8)
SIGNAL = 0.22
SCALE = (0.65, 0.65, 0.24, 0.30, 0.24)
ENCODER_SCALE = (1.5, 1.5, 0.7, 0.7, 0.7)
SCORE_SEED, SCORE_N = 27823, 1024


def freeze_protocol(root=OUTPUT):
    schedule = {k: ref.STAGES["lr_search"][k] for k in ref.RUN_FIELDS}
    schedule.update(dict(zip(("n_train", "n_val", "n_test"), SIZES, strict=True)),
                    clip_norm=5., adam_betas=(.9, .999), adam_eps=1e-8)
    protocol = {
        "tasks": TASKS, "models": MODELS, "parameters": {m: previous.COUNTS[m] for m in MODELS},
        "data_seeds": DATA_SEEDS, "init_seeds": previous.INIT_SEEDS, "dev_seed": DEV_SEED,
        "dev_inits": previous.DEV_INITS, "rates": previous.RATES, "schedule": schedule, "regularization": 0,
        "generator": {"latent_scale": SCALE, "encoder_scale": ENCODER_SCALE, "planted_triple": TRIPLE,
                      "pair_control": TRIPLE[:2], "signal_rms": SIGNAL, "normalizer": _normalizer()},
        "scorer": {"seed": SCORE_SEED, "rows": SCORE_N, "aggregation": "joint RMS over residual coordinates"},
        "runtime": {**previous._runtime(), "device": "cpu", "threads": 1, "workers": previous.WORKERS},
        "code_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in
                        (Path(__file__), Path(previous.__file__), Path(ref.__file__),
                         Path(base.__file__), Path(_mixed_difference.__code__.co_filename))},
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


def _latent(seed, n):
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(n, 10, generator=rng, dtype=torch.float64)


def _residual(task, latent):
    u = latent.tanh() / _normalizer()
    residual = latent.new_zeros((len(latent), 5))
    if task == "pair_control":
        each = SIGNAL / sqrt(2) * u[:, TRIPLE[:2]].prod(dim=-1)
        residual[:, 0] = residual[:, 2] = each
    else:
        assert task in TASKS[:2]
        residual[:, 0 if task == "center_triple" else 2] = SIGNAL * u[:, TRIPLE].prod(dim=-1)
    return residual


@lru_cache(maxsize=24)
def _data(task, seed):
    latent = _latent(seed, sum(SIZES))
    raw = latent.reshape(-1, 2, 5) * latent.new_tensor(SCALE)
    mu, cov = ref.decode_gaussian(raw)
    base_code = ref.encode_gaussian(mu.mean(1), cov.mean(1))
    mu_y, cov_y = ref.decode_gaussian(base_code + _residual(task, latent))
    inputs = tuple(v.split(SIZES) for v in (mu, cov))
    targets = tuple(v.split(SIZES) for v in (mu_y[:, None], cov_y[:, None]))
    return dict(zip(("train", "val", "test"), zip(zip(*inputs), zip(*targets)), strict=True))


@torch.inference_mode()
def _scores(model):
    bank = model.mgs_residual.banks[-1]
    rng = torch.Generator().manual_seed(SCORE_SEED)
    ratio = bank.center.new_tensor(SCALE) / bank.center.new_tensor(ENCODER_SCALE)
    a, b = [(torch.randn(SCORE_N, 10, generator=rng, dtype=torch.float64) * ratio.repeat(2))
            .tanh()[:, bank.indices] for _ in range(2)]
    factor = bank.factor()

    def group_values(points):
        delta = points.unsqueeze(-2) - bank.center
        projected = torch.einsum("ngki,gkij->ngkj", delta, factor)
        kernels = torch.exp(-0.5 * projected.square().sum(-1))
        return torch.einsum("ngk,gkc->ngc", kernels, bank.amplitude)

    raw = (_mixed_difference(group_values, a, b).square().sum(-1).mean(0) / 8).sqrt().tolist()
    assert len(raw) == 120 and all(torch.isfinite(torch.tensor(raw)))
    ordered = sorted(zip(bank.indices.tolist(), raw, strict=True), key=lambda r: (-r[1], r[0]))
    return [{"group": g, "score": s, "rank": rank} for rank, (g, s) in enumerate(ordered, 1)]


def _fit_job(args):
    output, protocol, stage, model, task, seed, init, lr = args
    with ref.float64():
        cfg = ref.RunCfg(model, task, seed, init, lr, **protocol["schedule"])
        identity = previous._identity(protocol, stage, cfg)
        path = output / stage / "fits" / f"{identity['key']}.pt"
        data = _data(task, seed)
        cached = path.exists()
        if not cached:
            previous._save_fit(path, identity, data, previous._fit(previous.make_model(model, init), cfg, data))
        evidence = previous._verify(path, identity, cfg, data)
        row = {"task": task, "model": model, "seed": seed, "init": init, "lr": lr, "key": identity["key"],
               "selected_step": evidence["selected_step"], "metrics": evidence["metrics"],
               "training_seconds": evidence["training_seconds"], "artifact": str(path.relative_to(output)),
               "sha256": sha256(path.read_bytes()).hexdigest()}
        if stage == "final":
            prediction, target = (evidence["splits"]["test"][k] for k in ("prediction", "target"))
            row["residual_rmse"] = ((ref.encode_gaussian(*prediction) - ref.encode_gaussian(*target))
                                    .square().sum(-1).mean().sqrt().item())
            if model == "III":
                bank_model = previous.make_model(model, init)
                bank_model.load_state_dict(evidence["checkpoint"], strict=True)
                row["triple_scores"] = _scores(bank_model)
        return row, cached


def _run(output, protocol, stage, jobs):
    assert json.loads((output / "protocol.json").read_text()) == protocol
    start, rows, reused = perf_counter(), [], 0
    with ProcessPoolExecutor(max_workers=previous.WORKERS, mp_context=get_context("spawn"),
                             initializer=_setup_torch) as pool:
        for row, cached in pool.map(_fit_job, [(output, protocol, stage, *job) for job in jobs]):
            rows.append(row)
            reused += cached
            if len(rows) % 15 == 0 or len(rows) == len(jobs):
                print(f"{stage}: {len(rows)}/{len(jobs)} verified ({reused} reused)", flush=True)
    assert len({r["key"] for r in rows}) == len(rows) == len(jobs)
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
    with ref.float64(), TemporaryDirectory() as name:
        x = _latent(SCORE_SEED, SCORE_N)
        b = _latent(SCORE_SEED + 1, SCORE_N)
        a, b = x[:, TRIPLE], b[:, TRIPLE]
        lower = _mixed_difference(lambda z: (z[..., 0] * z[..., 1]).tanh(), a, b)
        product_check = _mixed_difference(lambda z: z.prod(-1), a, b)
        torch.testing.assert_close(product_check, (a - b).prod(-1), atol=1e-13, rtol=1e-13)
        assert lower.abs().max().item() < 1e-14
        data = [_data(task, DEV_SEED) for task in TASKS]
        for split in data[0]:
            assert all(all(torch.equal(a, b) for a, b in zip(d[split][0], data[0][split][0], strict=True))
                       for d in data)
            for task, d in zip(TASKS, data, strict=True):
                (mu, cov), target = d[split]
                assert (torch.linalg.eigvalsh(target[1]) > 0).all()
                recovered = ref.encode_gaussian(*target).squeeze(1) - ref.encode_gaussian(mu.mean(1), cov.mean(1))
                offset = {"train": 0, "val": SIZES[0], "test": sum(SIZES[:2])}[split]
                latent = _latent(DEV_SEED, sum(SIZES))[offset:offset + len(mu)]
                torch.testing.assert_close(recovered, _residual(task, latent), atol=1e-13, rtol=1e-13)
        for model, task in product(MODELS, TASKS):
            cfg = ref.RunCfg(model, task, DEV_SEED, previous.DEV_INITS[0], 0.003,
                             **{**protocol["schedule"], "steps": 10})
            path = Path(name) / "smoke.pt"
            identity = previous._identity(protocol, "smoke", cfg)
            dataset = _data(task, DEV_SEED)
            previous._save_fit(path, identity, dataset,
                               previous._fit(previous.make_model(model, cfg.train_seed), cfg, dataset))
            previous._verify(path, identity, cfg, dataset)
    result = {"scorer_lower_order_max_error": lower.abs().max().item(), "smoke_fits": 9,
              "target_inverse_and_spd": "passed", "shared_inputs": "passed"}
    _write_json(output / "checks.json", result)
    return result


def tune(output, protocol):
    rows = _run(output, protocol, "development",
                list(product(MODELS, TASKS, (DEV_SEED,), previous.DEV_INITS, previous.RATES)))
    assert len(rows) == 54
    selected = {task: {} for task in TASKS}
    for task, model in product(TASKS, MODELS):
        ranked = []
        for lr in previous.RATES:
            fits = [r for r in rows if (r["task"], r["model"], r["lr"]) == (task, model, lr)]
            assert len(fits) == len(previous.DEV_INITS)
            ranked.append((mean(r["metrics"]["val"]["w2"] for r in fits), lr))
        selected[task][model] = min(ranked)[1]
    path = output / "development" / "selection.json"
    choice = {"protocol_id": protocol["id"], "selected": selected}
    if path.exists():
        assert json.loads(path.read_text()) == choice
    else:
        _write_json(path, choice)
    return choice


def run_final(output, protocol, choice):
    assert choice == json.loads((output / "development" / "selection.json").read_text())
    jobs = [(model, task, seed, init, choice["selected"][task][model])
            for model, task, seed, init in product(MODELS, TASKS, DATA_SEEDS, previous.INIT_SEEDS)]
    rows = _run(output, protocol, "final", jobs)
    assert len(rows) == 135
    return rows


def report(rows):
    from IPython.display import Markdown, display

    def values(task, model, seed=None):
        group = [r for r in rows if r["task"] == task and r["model"] == model
                 and (seed is None or seed == r["seed"])]
        assert len(group) == len(previous.INIT_SEEDS) * (len(DATA_SEEDS) if seed is None else 1)
        return tuple(mean(r["metrics"]["test"][key] for r in group) for key in previous.METRICS) + (
            mean(r["residual_rmse"] for r in group),)
    display(Markdown("### Геометрический прогноз"))
    base._table(("Задача", "Модель", "Параметры", "W₂²", "Центр", "Ковариация", "RMSE поправки"),
                [(t, m, previous.COUNTS[m], *(f"{v:.6f}" for v in values(t, m))) for t, m in product(TASKS, MODELS)])
    base._table(("Задача", "Seed", "III−II W₂²", "III−II центр", "III−II ковариация", "III−MLP W₂²"),
                [(t, s, *(f"{a-b:+.6f}" for a, b in zip(values(t, "III", s)[:3], values(t, "II", s)[:3])),
                  f"{values(t, 'III', s)[0] - values(t, 'Flat MLP', s)[0]:+.6f}")
                 for t, s in product(TASKS, DATA_SEEDS)])
    display(Markdown("### Восстановление тройки среди 120 групп III"))
    ranking, controls = [], []
    for row in rows:
        if row["model"] == "III":
            scores = row["triple_scores"]
            if row["task"] == "pair_control":
                controls.append((row["seed"], row["init"], f"{scores[0]['score']:.6f}"))
            else:
                truth = next(s for s in scores if tuple(s["group"]) == TRIPLE)
                ranking.append((row["task"], row["seed"], row["init"], truth["rank"], f"{truth['score']:.6f}"))
    base._table(("Задача", "Seed", "Инициализация", "Ранг (0,2,8)", "Сила"), ranking)
    base._table(("Seed", "Инициализация", "Максимум тройного вклада на попарном контроле"), controls)
