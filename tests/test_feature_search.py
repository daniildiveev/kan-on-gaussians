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
                tracking=tracking, monitoring=MonitorSettings())
    rows = grid.run(**args)
    assert grid.size == rows.run_id.nunique() == 4
    assert set(grid.compare(rows).feature_set) == {"ambient", "all"}
    report, paired = grid.feature_reports(rows)
    assert set(report.n_features) == {1, 4}
    assert len(paired) == 2
    for row in rows.itertuples():
        run = MlflowClient(tracking_uri=tracking.tracking_uri).get_run(row.run_id)
        assert len(json.loads(run.data.params["feature_columns"])) == row.n_features
        assert not any(metric.startswith("test_") for metric in run.data.metrics)
    assert list(grid.run(**args).run_id) == list(rows.run_id)
