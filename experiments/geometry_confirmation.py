"""Fresh-seed confirmation for all seven frozen Gaussian geometry arms."""

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
import math
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
import time

import torch
from torch.func import functional_call, vmap

from experiments import geometry_benchmark as original
from experiments import geometry_reference as ref
from src.experiment import _atomic_file, _digest, _runtime, _write_json

OUTPUT = Path("outputs/study/geometry_confirmation")
DATA_SEEDS = (6000, 7000, 8000)
TRAIN_SEEDS = (11, 29, 47)
MODELS = (original.MODEL, *ref.MODELS)
HISTORICAL = Path("outputs/study/geometry_benchmark/1bf2a83d7b3d1e7f")
TRAINER_NAMES = {"_per_run", "stack_split", "make_batched_forward", "init_params", "unstack_model", "train_group"}
WORK_ORDER = ("Anchor / BW", "Spline / BW", "MLP edges / BW", original.MODEL,
              "Spline / mean", "MLP edges / mean", "Flat MLP")
SCORE_FIELDS = ("w2", "mean_error", "cov_error")


@dataclass(frozen=True)
class Confirmation:
    output: Path
    notebook: Path
    protocol: dict

    @property
    def id(self):
        return _digest(self.protocol)


def _read(path):
    return json.loads(path.read_text())


def _build_model(name, bary_iters):
    return original.GeometryMGSKAN() if name == original.MODEL else ref.build_model(name, bary_iters)


def _trainer_source(notebook):
    return "".join(_read(notebook)["cells"][19]["source"])


def _trainer(notebook, penalty):
    """Reuse the benchmark update loop and add the MGS-KAN penalty before clipping."""
    nodes = [n for n in ast.parse(_trainer_source(notebook)).body
             if isinstance(n, ast.FunctionDef) and n.name in TRAINER_NAMES]
    assert {n.name for n in nodes} == TRAINER_NAMES
    train = next(n for n in nodes if n.name == "train_group")
    loop = next(n for n in train.body if isinstance(n, ast.While))
    index = next(i for i, n in enumerate(loop.body)
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "grads")
    extra = ast.parse(
        "if c0.model == 'MGS-KAN':\n"
        "    amplitudes = params['mgs_residual.pair_amplitude']\n"
        "    loss = loss + penalty * torch.linalg.vector_norm(amplitudes.flatten(2), dim=-1).sum(-1)\n"
    ).body[0]
    loop.body.insert(index, ast.copy_location(extra, loop.body[index]))
    scope = {"torch": torch, "copy": copy, "time": time, "functional_call": functional_call,
             "vmap": vmap, "build_model": _build_model, "penalty": penalty,
             **{key: getattr(ref, key) for key in ("make_dataset", "DTYPE", "CURVE_COLS", "sync", "w2_parts", "w2sq")}}
    unit = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(unit, f"{notebook}:geometry trainer", "exec"), scope)
    return scope["train_group"], scope["unstack_model"]


def freeze_protocol(root=OUTPUT, notebook=original.REFERENCE, archive=original.ARCHIVE,
                    historical=HISTORICAL):
    notebook, archive, historical = map(Path, (notebook, archive, historical))
    prior = _read(historical / "protocol.json")
    selection = _read(historical / "selected_lr.json")
    assert selection["protocol_id"] == _digest(prior)
    rates = {arm: {} for arm in MODELS}
    rates[original.MODEL] = selection["selected"]
    for row in _read(archive / "selected_lr.json")["selection"]:
        rates[row["model"]][row["task"]] = row["lr"]
    assert all(set(values) == set(ref.TASKS) for values in rates.values())
    previous_data = {row["data_seed"] for line in (archive / "runs.jsonl").read_text().splitlines()
                     if line.strip() and (row := json.loads(line))["status"] == "done"}
    assert not previous_data.intersection(DATA_SEEDS)
    assert not set(prior["stage_d"]["data_seeds"]).intersection(DATA_SEEDS)
    params = {"MGS-KAN": 2515, "Flat MLP": 2483, "Anchor / BW": 162,
              "Spline / mean": 2520, "Spline / BW": 2520, "MLP edges / mean": 2520, "MLP edges / BW": 2520}
    schedule = {key: prior["stage_d"][key] for key in ref.RUN_FIELDS}
    schedule.update(clip_norm=prior["clip_norm"], adam_betas=prior["adam_betas"], adam_eps=prior["adam_eps"])
    protocol = {
        "data_seeds": DATA_SEEDS, "train_seeds": TRAIN_SEEDS, "tasks": ref.TASKS, "models": MODELS,
        "learning_rates": rates, "parameters": params, "schedule": schedule,
        "penalty": prior["group_penalty"], "single_kernels": prior["single_kernels"],
        "pair_kernels": prior["pair_kernels"], "checkpoint_every": prior["stage_d"]["save_every"],
        "checkpoint_rule": "earliest minimum unpenalized validation W2 squared, including step 0",
        "batch_rng": "CPU generator seeded by train_seed + 10000, draws with replacement",
        "runtime": {**_runtime(), "device": "cpu", "dtype": "float64", "workers": 5, "threads_per_worker": 1},
        "historical_protocol_id": selection["protocol_id"],
        "code_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), Path(original.__file__), Path(ref.__file__))},
        "reference_trainer_sha256": sha256(_trainer_source(notebook).encode()).hexdigest(),
    }
    protocol = json.loads(json.dumps(protocol))
    output = Path(root) / _digest(protocol)[:16]
    if (output / "protocol.json").exists():
        assert _read(output / "protocol.json") == protocol
    else:
        _write_json(output / "protocol.json", protocol)
    return Confirmation(output, notebook, protocol)


def _configure():
    torch.set_num_threads(1)


def _configs(study, name):
    p = study.protocol
    for t, d, s in product(p["tasks"], p["data_seeds"], p["train_seeds"]):
        yield ref.RunCfg(name, t, d, s, p["learning_rates"][name][t], **p["schedule"])


def _fit_path(study, cfg):
    slug = "mgs-kan" if cfg.model == original.MODEL else ref.MODEL_SLUG[cfg.model]
    return study.output / "fits" / slug / f"{original._run_key(cfg)}.pt"


def _identity(study, cfg):
    return original._identity(study, cfg)


@torch.inference_mode()
def _verify(study, cfg, path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["identity"] == _identity(study, cfg)
    eval_steps = list(range(0, cfg.steps + 1, cfg.eval_every))
    history = saved["history"]
    assert saved["eval_steps"] == eval_steps
    assert history.shape == (len(eval_steps), len(ref.CURVE_COLS)) and torch.isfinite(history).all()
    step = eval_steps[history[:, 3].argmin().item()]
    assert saved["selected_step"] == step
    model = _build_model(cfg.model, cfg.bary_iters)
    model.load_state_dict(saved["checkpoint"], strict=True)
    count = ref.count_params(model)
    assert count == study.protocol["parameters"][cfg.model]
    data = ref.make_dataset(cfg.task, cfg.data_seed, cfg.n_train, cfg.n_val, cfg.n_test)
    assert set(saved["splits"]) == set(saved["metrics"]) == set(data)
    for split, (inputs, target) in data.items():
        heldout = saved["splits"][split]
        for x, y in zip(target, heldout["target"], strict=True):
            assert torch.equal(x, y)
        for x, y in zip(model(*inputs), heldout["prediction"], strict=True):
            torch.testing.assert_close(x, y, rtol=original.RTOL, atol=original.ATOL)
        recomputed = original._metrics(heldout["prediction"], heldout["target"])
        for name, value in recomputed.items():
            assert math.isclose(value, saved["metrics"][split][name], rel_tol=original.RTOL, abs_tol=original.ATOL)
    assert math.isclose(saved["metrics"]["val"]["w2"], history[eval_steps.index(step), 3].item(),
                        rel_tol=original.RTOL, abs_tol=original.ATOL)
    seconds = saved["seconds_group"]
    assert math.isfinite(seconds) and seconds >= 0
    return {"key": saved["identity"]["key"], "task": cfg.task, "model": cfg.model,
            "data_seed": cfg.data_seed, "train_seed": cfg.train_seed, "lr": cfg.lr,
            "parameters": count, "selected_step": step, "metrics": saved["metrics"],
            "seconds_group": seconds, "training_seconds": seconds / len(list(_configs(study, cfg.model))),
            "device": study.protocol["runtime"]["cpu"], "provenance": "fresh confirmation",
            "artifact": str(path.relative_to(study.output)), "sha256": sha256(path.read_bytes()).hexdigest()}


@torch.inference_mode()
def _publish(study, cfg, path, model, run, index):
    data = ref.make_dataset(cfg.task, cfg.data_seed, cfg.n_train, cfg.n_val, cfg.n_test)
    splits, metrics = {}, {}
    for name, (inputs, target) in data.items():
        prediction = model(*inputs)
        splits[name] = {"target": target, "prediction": prediction}
        metrics[name] = original._metrics(prediction, target)
    original._save_pt(path, {"identity": _identity(study, cfg), "checkpoint": model.state_dict(),
                             "selected_step": int(run["best_step"][index]), "eval_steps": run["eval_steps"],
                             "history": run["curve"][index].clone(), "splits": splits, "metrics": metrics,
                             "seconds_group": run["seconds"]})


def _run_arm(study, name):
    with ref.float64():
        cfgs = list(_configs(study, name))
        paths = [_fit_path(study, cfg) for cfg in cfgs]
        group_snapshot = paths[0].parent / "training.partial.pt"
        rows = {}
        for cfg, path in zip(cfgs, paths, strict=True):
            if path.exists():
                rows[path] = _verify(study, cfg, path)
        reused = len(rows)
        if reused != len(cfgs):
            identities = [_identity(study, cfg) for cfg in cfgs]
            resume = None
            if group_snapshot.exists():
                partial = torch.load(group_snapshot, weights_only=True, map_location="cpu")
                assert partial["identities"] == identities
                resume = partial["state"]

            def checkpoint(snapshot):
                original._save_pt(group_snapshot, {"identities": identities, "state": snapshot})

            train, unstack = _trainer(study.notebook, study.protocol["penalty"])
            completed = train(cfgs, torch.device("cpu"), resume=resume,
                              save_every=study.protocol["checkpoint_every"], on_checkpoint=checkpoint)
            assert completed["status"] == "finished" and completed["step"] == cfgs[0].steps
            assert (completed["first_bad"] < 0).all()
            for i, (cfg, path) in enumerate(zip(cfgs, paths, strict=True)):
                if path not in rows:
                    model = unstack(completed["template"], completed["best_params"], i)
                    _publish(study, cfg, path, model, completed, i)
                    rows[path] = _verify(study, cfg, path)
        group_snapshot.unlink(missing_ok=True)
        return name, [rows[path] for path in paths], reused


def run_confirmation(study):
    rows, reused = {}, 0
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=study.protocol["runtime"]["workers"], mp_context=get_context("spawn"),
                             initializer=_configure) as pool:
        pending = [pool.submit(_run_arm, study, name) for name in WORK_ORDER]
        for job in as_completed(pending):
            name, fits, cached = job.result()
            rows[name] = fits
            reused += cached
            print(f"{name}: {len(fits)}/27 verified ({cached} reused)", flush=True)
    fits = [row for name in MODELS for row in rows[name]]
    expected = set(product(ref.TASKS, MODELS, DATA_SEEDS, TRAIN_SEEDS))
    assert len(fits) == len(expected) == 189
    assert set((r["task"], r["model"], r["data_seed"], r["train_seed"]) for r in fits) == expected
    assert set(p for p in (study.output / "fits").glob("*/*.pt")) == {study.output / r["artifact"] for r in fits}
    manifest = {"protocol_id": study.id, "fits": fits}
    path = study.output / "manifest.json"
    if path.exists():
        assert _read(path) == manifest
    else:
        _write_json(path, manifest)
    print(f"189/189 confirmation fits verified ({reused} reused) in {time.perf_counter() - start:.1f}s", flush=True)
    return fits


def report(study, fits):
    from IPython.display import Markdown, display

    original._fit_csv(study.output, fits)
    paired_rows, rankings, shifted = [], {}, []
    for task in ref.TASKS:
        ordered = sorted(MODELS, key=lambda model: original._aggregates(fits, task, model)[0])
        rankings[task] = ordered
        display(Markdown(f"### {task.capitalize()}"))
        original._table(("Model", "Parameters", "Test W₂²", "Center", "Covariance"), [
            (model, study.protocol["parameters"][model],
             *(f"{v:.7f}" for v in original._aggregates(fits, task, model))) for model in ordered
        ])
        for comparator in MODELS[1:]:
            differences = [original._aggregates(fits, task, original.MODEL, seed=s)[0] -
                           original._aggregates(fits, task, comparator, seed=s)[0] for s in DATA_SEEDS]
            totals = [a - b for a, b in zip(original._aggregates(fits, task, original.MODEL),
                                            original._aggregates(fits, task, comparator), strict=True)]
            paired_rows.append((task, comparator, *differences, *totals))
        for model in MODELS:
            shifted.append((task, model, *(f"{original._aggregates(fits, task, model, split)[0]:.7f}"
                                           for split in ref.OOD_VARIANTS)))
    headers = ("Task", "Reference", *(f"Seed {s}" for s in DATA_SEEDS), "Mean Δ", "Center Δ", "Covariance Δ")
    csv_path = study.output / "paired_differences.csv"
    with _atomic_file(csv_path) as temp:
        with temp.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(paired_rows)
    display(Markdown("### Paired W₂² differences"))
    original._table(headers, [(*r[:2], *(f"{v:+.7f}" for v in r[2:])) for r in paired_rows])
    display(Markdown("### Distribution shift"))
    original._table(("Task", "Model", "ID redraw", "Mean shift", "Covariance shift", "Both shifts"), shifted)
    display(Markdown("### Timing"))
    original._table(("Model", "Device", "Seconds for group of 27", "Amortized seconds per fit"), [
        (m, next(r["device"] for r in fits if r["model"] == m),
         f"{next(r['seconds_group'] for r in fits if r['model'] == m):.2f}",
         f"{mean(r['training_seconds'] for r in fits if r['model'] == m):.2f}") for m in MODELS
    ])
    verdict = []
    for task in ref.TASKS:
        flat = next(r for r in paired_rows if r[:2] == (task, "Flat MLP"))
        wins = sum(v < 0 for v in flat[2:5])
        ordered = rankings[task]
        verdict.append(f"- **{task.capitalize()}** ranks {ordered.index(original.MODEL) + 1}/7 for MGS-KAN. "
                       f"{ordered[0]} leads. MGS-KAN has lower W₂² than Flat MLP on {wins}/3 splits "
                       f"(mean difference {flat[5]:+.7f}).")
    display(Markdown("### Fresh-seed verdict\n\n" + "\n".join(verdict) +
                     "\n\nEach comparison averages the three training initializations within each fresh data seed. "
                     "CPU groups were trained concurrently. Historical GPU timings remain in the preceding section."))
