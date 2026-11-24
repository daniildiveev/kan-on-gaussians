import json
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import dataset_class, load_dataset

pytest.importorskip("mlflow")

from src.config import MonitorSettings, ResourceSettings, TrackingSettings, TrainingSettings
from src.experiments.features import FeaturePolicy
from src.experiments.search import ValidationGrid
from src.models import make_model, parameter_count


@pytest.mark.parametrize("name,count", [("airfoil", 31), ("power_plant", 15)])
def test_exhaustive_feature_sets_and_parameter_matching(name, count):
    grid = ValidationGrid.preset("features", datasets=(name,))
    columns = dataset_class(name).columns
    expected = {group for k in range(1, len(columns)+1) for group in combinations(columns, k)}
    assert set(grid.subsets(name).values()) == expected
    assert grid.size == 12 * 3 * count
    for _, _, selected, stage, settings in grid.plan():
        if "mgs_kan" in stage.models:
            settings = settings.model_copy(update={"features": len(selected)})
            kan = parameter_count(make_model("mgs_kan", settings))
            width = settings.mlp_width
            size = parameter_count(make_model("mlp", settings))
            for other in (max(1, width - 1), width + 1):
                nearby = settings.model_copy(update={"mlp_width": other})
                assert abs(size - kan) <= abs(parameter_count(make_model("mlp", nearby)) - kan)


def test_grouped_features_cover_all_registered_tasks():
    grid = ValidationGrid.preset("features")
    for dataset in grid.datasets:
        subsets = grid.subsets(dataset)
        assert subsets["all"] == dataset_class(dataset).columns
        assert len(subsets) == len(set(subsets.values()))
    assert len(grid.subsets("sarcos")) == 7
    assert len(grid.subsets("superconductivity")) == 9
    with pytest.raises(ValueError, match="Exhaustive"):
        FeaturePolicy(exhaustive=True).subsets("superconductivity")


def test_projection_preserves_rows_and_scaling(tmp_path):
    values = np.arange(40)
    frame = pd.DataFrame({key: values + i for i, key in enumerate(["AT", "V", "AP", "RH", "PE"])})
    frame.to_csv(tmp_path / "power_plant.csv", index=False)
    full = load_dataset("power_plant", root=tmp_path, fold=1)
    subset = load_dataset("power_plant", root=tmp_path, fold=1, feature_columns=("RH", "AT"))
    for split in ["train", "val", "test"]:
        x, y = getattr(subset, split).tensors
        X, Y = getattr(full, split).tensors
        torch.testing.assert_close(x, X[:, [3, 0]])
        torch.testing.assert_close(subset.inverse_target(y), full.inverse_target(Y))


def test_feature_selection_through_workers_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    from mlflow import MlflowClient
    from src.experiments import ExperimentRunner
    from src.experiments.failures import WorkerFailed
    from src.monitoring.client import MonitorSession

    events = []

    def receive(session, event):
        events.append(event)
        return 204

    monkeypatch.setattr(MonitorSession, "_post", receive)
    raw = np.arange(150).reshape(30, 5)
    np.savetxt(tmp_path / "power_plant.csv", raw, delimiter=",", header="AT,V,AP,RH,PE", comments="")
    profile = ValidationGrid.preset("features", datasets=("power_plant",))
    grid = ValidationGrid(
        datasets=profile.datasets, feature_policies={"power_plant": FeaturePolicy(groups={"ambient": ("AT",)})},
        stages=profile.stages[:1], match_mlp_budget=True, seeds=((11, 101),),
        training=TrainingSettings(epochs=1),
    )
    tracking = TrackingSettings(tracking_uri=(tmp_path/"mlruns").as_uri(), experiment_name="subsets")
    args = dict(version="test", device="cpu", root=tmp_path, resources=ResourceSettings(cpu_threads=1),
                tracking=tracking, monitoring=MonitorSettings(endpoint="https://monitor.test/v1/events",
                                                              token="test"))

    worker = ExperimentRunner._worker

    def fail_second_series(self, request):
        if request["action"] == "trial" and request["feature_columns"] == ("AT",):
            raise WorkerFailed("worker stopped", "cuda_oom", "failed-run")
        return worker(self, request)

    with monkeypatch.context() as patch:
        patch.setattr(ExperimentRunner, "_worker", fail_second_series)
        with pytest.raises(RuntimeError, match="worker stopped"):
            grid.run(**args)
    terminal, = [event for event in events if event.terminal]
    assert terminal.kind == "failed" and terminal.reason == "cuda_oom"
    assert terminal.completed == 2 and terminal.total == 4
    assert "features-ambient" in terminal.series_id and terminal.run_id == "failed-run"
    assert terminal.first_result.model == "mlp"
    assert [point.percent for point in terminal.milestones] == [25, 50]

    events.clear()
    timer = iter([100.0, 165.9, 200.0, 203.1, 300.0, 310.0])
    monkeypatch.setattr("src.experiments.search.monotonic", lambda: next(timer))
    rows = grid.run(**args)
    terminal, = [event for event in events if event.terminal]
    assert len({event.session_id for event in events}) == 1
    assert terminal.kind == "finished" and terminal.grid_id == "power_plant · test"
    assert terminal.completed == terminal.total == 4 and terminal.elapsed_seconds == 65
    assert [point.percent for point in terminal.milestones] == [75]
    first_new = rows[rows.feature_set == "ambient"].iloc[0]
    assert terminal.first_result.run_id == first_new.run_id
    assert terminal.first_result.val_rmse == first_new.val_rmse
    assert grid.size == rows.run_id.nunique() == 4
    assert set(grid.compare(rows).feature_set) == {"ambient", "all"}
    report, paired = grid.feature_reports(rows)
    assert set(report.n_features) == {1, 4}
    assert len(paired) == 2
    for row in rows.itertuples():
        run = MlflowClient(tracking_uri=tracking.tracking_uri).get_run(row.run_id)
        assert len(json.loads(run.data.params["feature_columns"])) == row.n_features
        assert not any(metric.startswith("test_") for metric in run.data.metrics)
    events.clear()
    assert list(grid.run(**args).run_id) == list(rows.run_id)
    resumed, = [event for event in events if event.terminal]
    assert resumed.session_id != terminal.session_id
    assert resumed.kind == "finished" and resumed.completed == 4 and resumed.elapsed_seconds == 3
    assert resumed.first_result is None and resumed.milestones is None

    # Only the later series is cached. Its runs already count toward the baseline.
    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    for run_id in rows.loc[rows.feature_set == "all", "run_id"]:
        client.delete_run(run_id)
    events.clear()
    grid.run(**args)
    final = events[-1]
    assert final.completed == 4 and [point.percent for point in final.milestones] == [75]
    assert "features-all" in final.milestones[0].series_id
