from collections import defaultdict

import pytest

pytest.importorskip("mlflow")

from src.config import MonitorSettings, ResourceSettings, TrackingSettings
from src.data import dataset_class
from src.experiments import ExperimentRunner
from src.experiments.search import ValidationGrid
from src.models import make_model, parameter_count
from src.tracking.mlflow import MatchingTrials


def test_parameter_budgets_match_real_models_and_legacy_compact_pairs():
    grid = ValidationGrid.preset("parameter_budget")
    assert grid.size == 108
    assert grid.datasets == ("airfoil", "power_plant") and grid.reuse_matching_validation
    budgets = defaultdict(dict)
    compact = {}
    for dataset, label, columns, stage, settings in grid.plan():
        assert label == "all" and columns is None
        opts = settings.model_copy(update={"features": len(dataset_class(dataset).columns)})
        counts = {m: parameter_count(make_model(m, opts)) for m in stage.models}
        if stage.history_only:
            compact[dataset, stage.name] = tuple(counts.values())
        else:
            budgets[dataset, stage.parameter_budget].update(counts)

    assert compact == {
        ("airfoil", "stage-1"): (106, 111), ("airfoil", "stage-2"): (181, 186),
        ("airfoil", "stage-3"): (346, 351), ("power_plant", "stage-1"): (79, 77),
        ("power_plant", "stage-2"): (121, 125), ("power_plant", "stage-3"): (229, 233),
    }
    assert set(budgets) == {(d, b) for d in grid.datasets for b in (1000, 4000, 16000)}
    for (_, target), counts in budgets.items():
        assert set(counts) == {"mlp", "mgs_kan", "deep_mlp", "resnet"}
        assert all(abs(n - target) <= 0.05 * target for n in counts.values())
        assert max(counts.values()) <= 1.05 * min(counts.values())


def test_missing_compact_history_stops_before_training(monkeypatch, tmp_path):
    grid = ValidationGrid.preset("parameter_budget", datasets=("airfoil",))
    def empty_history(plan, *_):
        return {series.key: MatchingTrials("unused", {}) for _, _, series in plan}

    monkeypatch.setattr(ValidationGrid, "_matching_history", staticmethod(empty_history))
    monkeypatch.setattr(ExperimentRunner, "run_series", lambda *a, **kw: pytest.fail("Started training"))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MGS_KAN_LOG_DIR", str(tmp_path / "logs"))
    with pytest.raises(ValueError, match="0/6 matching compact runs"):
        grid.run(version="test", device="cpu", resources=ResourceSettings(), monitoring=MonitorSettings(),
                 tracking=TrackingSettings(tracking_uri=(tmp_path / "mlruns").as_uri()))
