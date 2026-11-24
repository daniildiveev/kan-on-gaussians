"""Paired distribution shifts using the four frozen geometry-order checkpoints."""

from hashlib import sha256
import json
from pathlib import Path

import torch

from experiments import geometry_benchmark as base, geometry_order as order, geometry_reference as ref
from src.experiment import _digest, _setup_torch, _write_json

OUTPUT = Path("outputs/study/geometry_shift")
SHIFTS = {"ood_id": "ID redraw", "ood_mean": "Center shift",
          "ood_cov": "Covariance shift", "ood": "Both shifts"}


@torch.inference_mode()
def run_shift(source, root=OUTPUT):
    _setup_torch()
    source = Path(source)
    upstream = json.loads((source / "protocol.json").read_text())
    manifest_path = source / "final" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["fits"]) == len({r["key"] for r in manifest["fits"]}) == 108
    assert manifest["protocol_id"] == upstream["id"]
    assert upstream["id"] == _digest({k: v for k, v in upstream.items() if k != "id"})
    protocol = {
        "source_protocol": upstream["id"], "source_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "shifts": ref.OOD_VARIANTS, "draw_seed": "data_seed+4",
        "code_sha256": {file.name: sha256(file.read_bytes()).hexdigest() for file in
                        (Path(__file__), Path(ref.__file__), Path(order.__file__))},
    }
    protocol = json.loads(json.dumps(protocol))
    protocol["id"] = _digest(protocol)
    output = Path(root) / protocol["id"][:16]
    if (output / "protocol.json").exists():
        assert json.loads((output / "protocol.json").read_text()) == protocol
    else:
        _write_json(output / "protocol.json", protocol)
    results, reused = [], 0
    with ref.float64():
        for row in manifest["fits"]:
            checkpoint = source / row["artifact"]
            assert sha256(checkpoint.read_bytes()).hexdigest() == row["sha256"]
            cfg = ref.RunCfg(row["model"], row["task"], row["data_seed"], row["train_seed"], row["lr"],
                             **upstream["schedule"])
            dataset = ref.make_dataset(cfg.task, cfg.data_seed, cfg.n_train, cfg.n_val, cfg.n_test)
            source_identity = order._identity(upstream, "final", cfg)
            original = order._verify(checkpoint, source_identity, cfg, dataset)
            assert original["metrics"] == row["metrics"]
            model = order.make_model(cfg.model, cfg.train_seed)
            model.load_state_dict(original["checkpoint"], strict=True)
            splits, metrics = {}, {}
            for name in SHIFTS:
                inputs, target = dataset[name]
                prediction = model(*inputs)
                splits[name] = {"prediction": prediction, "target": target}
                metrics[name] = base._metrics(prediction, target)
            path = output / "fits" / f"{row['key']}.pt"
            if path.exists():
                saved = torch.load(path, weights_only=True, map_location="cpu")
                assert saved["source_sha256"] == row["sha256"] and saved["metrics"] == metrics
                assert saved["protocol_id"] == protocol["id"] and set(saved["splits"]) == set(splits)
                for name in splits:
                    for field in ("prediction", "target"):
                        assert all(torch.equal(a, b) for a, b in
                                   zip(saved["splits"][name][field], splits[name][field], strict=True))
                reused += 1
            else:
                base._save_pt(path, {"protocol_id": protocol["id"], "source_sha256": row["sha256"],
                                     "metrics": metrics, "splits": splits})
            results.append({**row, "metrics": metrics, "source_sha256": row["sha256"],
                            "artifact": str(path.relative_to(output)), "sha256": sha256(path.read_bytes()).hexdigest()})
    assert {path.stem for path in (output / "fits").glob("*.pt")} == {row["key"] for row in results}
    result = {"protocol_id": protocol["id"], "fits": results}
    path = output / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        _write_json(path, result)
    base._fit_csv(output, results)
    print(f"108 checkpoints, 432 shift evaluations verified ({reused} cache files reused).")
    return output, results


def report_shift(rows):
    from IPython.display import Markdown, display

    display(Markdown("### Прогноз при сдвигах распределения"))
    table = []
    paired = []
    for task in ref.TASKS:
        for model in order.MODELS:
            initial = base._aggregates(rows, task, model, "ood_id")
            for split, label in SHIFTS.items():
                totals = base._aggregates(rows, task, model, split)
                table.append((task, model, label, *(f"{v:.6f}" for v in totals), f"{totals[0]-initial[0]:+.6f}"))
        for split, label in SHIFTS.items():
            values = [base._aggregates(rows, task, "III", split, s)[0]
                      - base._aggregates(rows, task, "II", split, s)[0] for s in order.DATA_SEEDS]
            paired.append((task, label, *(f"{v:+.6f}" for v in values)))
    base._table(("Задача", "Модель", "Условие", "W₂²", "Центр", "Ковариация", "Δ W₂² к ID"), table)
    display(Markdown("### Парная разность W₂², III минус II"))
    base._table(("Задача", "Условие", *map(str, order.DATA_SEEDS)), paired)
