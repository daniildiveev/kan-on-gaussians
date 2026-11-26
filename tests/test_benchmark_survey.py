import numpy as np
import pytest

pytest.importorskip("mlflow")

from src.config import MonitorSettings, ResourceSettings, TrackingSettings, TrainingSettings
from src.experiments.search import BENCHMARKS, ValidationGrid
from src.models import make_model, parameter_count


def test_survey_waves_cover_every_task_and_family():
    grid = ValidationGrid.preset("benchmark_survey")
    assert grid.size == 72
    for wave, width, depth, single, pair in [
        (1, 128, 4, 7, 4), (2, 256, 6, 16, 8),
    ]:
        part = grid.wave(wave)
        assert part.size == 36
        assert part.datasets == BENCHMARKS and part.folds == (0,) and part.seeds == ((11, 101),)
        for dataset, label, columns, stage, settings in part.plan():
            assert label == "all" and columns is None
            assert set(stage.models) == {"mlp", "mgs_kan", "deep_mlp", "resnet"}
            assert (settings.deep_width, settings.deep_depth) == (width, depth)
            assert (settings.single_kernels, settings.pair_kernels) == (single, pair)
            if dataset == "power_plant":
                opts = settings.model_copy(update={"features": 4})
                budget = parameter_count(make_model("mgs_kan", opts))
                best = abs(parameter_count(make_model("mlp", opts)) - budget)
                for delta in [-1, 1]:
                    other = opts.model_copy(update={"mlp_width": opts.mlp_width + delta})
                    assert best <= abs(parameter_count(make_model("mlp", other)) - budget)


def test_survey_reuses_only_matching_validation_checkpoints_across_series(tmp_path, monkeypatch):
    from dataclasses import replace

    from src.data import load_dataset
    from src.experiments import ExperimentRunner
    from src.tracking import MlflowTracker

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    np.savetxt(tmp_path / "power_plant.csv", np.random.default_rng(4).normal(size=(60, 5)),
               header="AT,V,AP,RH,PE", delimiter=",", comments="")
    training = TrainingSettings(epochs=2, batch_size=16, learning_rate=0.001)
    tracking = TrackingSettings(tracking_uri=(tmp_path / "mlruns").as_uri(), experiment_name="survey")
    resources = ResourceSettings(cpu_threads=1, max_parallel_trials=2)
    wave = ValidationGrid.preset("benchmark_survey", datasets=("power_plant",)).wave(1)
    wave = wave.model_copy(update={"training": training})
    settings = next(wave.plan())[-1]
    args = dict(device="cpu", monitoring=MonitorSettings(), tracking=tracking, resources=resources, root=tmp_path)
    tracker = MlflowTracker(tracking)
    data = load_dataset("power_plant", root=tmp_path, fold=0)
    legacy = ExperimentRunner(training=training, resources=resources, tracking=tracking,
                              model_settings=settings.model_copy(update={"deep_depth": 2}))
    old = legacy.run(series_id="unrelated-v1", dataset="power_plant", models=["mlp", "deep_mlp", "mgs_kan"],
                     folds=[0], seeds=[(11, 101)], device="cpu", root=tmp_path, evaluation="validation",
                     feature_columns=tuple(data.columns))
    ids = {r.trial.model: r.run_id for r in old.results}
    tracker.client.set_tag(ids["mgs_kan"], "mgs_kan.checkpoint_file", "missing.pt")

    runs = wave.run(version="test", **args).set_index("model")
    assert runs.reused.to_dict() == {"mlp": True, "mgs_kan": False, "deep_mlp": False, "resnet": False}
    assert runs.at["mlp", "run_id"] == ids["mlp"]
    assert runs.at["deep_mlp", "run_id"] != ids["deep_mlp"]
    assert runs.attrs["reused"] == 1 and runs.run_id.nunique() == 4
    for row in runs.itertuples():
        history = tracker.client.get_metric_history(row.run_id, "val_r2")
        assert row.val_r2 == next(point.value for point in history if point.step == row.best_epoch)
        assert not any(k.startswith("test_") for k in tracker.client.get_run(row.run_id).data.metrics)

    monkeypatch.setattr(ExperimentRunner, "_worker", lambda *a, **kw: pytest.fail("Unexpected retraining"))
    resumed = wave.run(version="new-series-name", **args).set_index("model")
    assert resumed.reused.all() and runs.run_id.to_dict() == resumed.run_id.to_dict()

    series = next(wave._series_plan("test", "cpu", resources))[-1]
    for changed in [
        {"seeds": ((11, 202),)}, {"seeds": ((22, 101),)},
        {"training": training.model_copy(update={"learning_rate": 0.01})},
        {"training": training.model_copy(update={"batch_size": 8})},
        {"training": training.model_copy(update={"epochs": 3})},
    ]:
        assert not tracker.matching_validation_trials(series.model_copy(update=changed), data).completed
    reordered = data.select_features(tuple(reversed(data.columns)))
    assert not tracker.matching_validation_trials(series, reordered).completed
    assert not tracker.matching_validation_trials(series, replace(data, target_column="different")).completed
    assert not tracker.matching_validation_trials(series, replace(data, source_sha256="a" * 64)).completed
    client = tracker.client
    for run_id in runs.run_id:
        client.set_tag(run_id, "mgs_kan.evaluation", "test")
    client.set_tag(ids["mlp"], "mgs_kan.evaluated", "false")
    assert not tracker.matching_validation_trials(series, data).completed
