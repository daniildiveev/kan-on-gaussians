import os
import signal
import subprocess
from unittest.mock import Mock

import pytest

pytest.importorskip("mlflow")

from src.config import ModelSettings, ResourceSettings, TrackingSettings, TrainingSettings
from src.data import CsvSpec
from src.experiments import ExperimentRunner, ExperimentSeries, SeriesSummary, Trial, TrialResult


@pytest.fixture
def local_series(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    table = tmp_path / "table.csv"
    table.write_text("x,target\n" + "".join(f"{i},{2*i}\n" for i in range(40)))
    settings = TrackingSettings(tracking_uri=(tmp_path / "mlruns").as_uri(), experiment_name="series")
    runner = ExperimentRunner(training=TrainingSettings(epochs=1), tracking=settings)
    args = dict(series_id="local-series", dataset=CsvSpec(table.name, "target"), root=tmp_path,
                models=["mlp", "mgs_kan"], folds=[1], seeds=[(17, 42)], device="cpu")
    return runner, args


def test_series_in_fresh_processes_and_resume(local_series, tmp_path, monkeypatch):
    runner, args = local_series
    work = tmp_path / "kernel"
    work.mkdir()
    monkeypatch.chdir(work)
    # A same-named module in the kernel directory must not shadow the worker's checkout.
    (work / "src.py").write_text("raise RuntimeError('wrong src')\n")
    original_threads = os.environ.get("OMP_NUM_THREADS")
    summary = runner.run(**args, max_runs=2)
    assert [r.trial.model for r in summary.results] == args["models"]
    assert summary.pending == ()
    assert os.environ.get("OMP_NUM_THREADS") == original_threads
    for result in summary.results:
        run = runner.tracker.client.get_run(result.run_id)
        assert (run.data.params["fold"], run.data.params["model.init_seed"],
                run.data.params["shuffle_seed"]) == ("1", "17", "42")
        assert run.data.tags["mgs_kan.evaluated"] == "true"
        assert result.test_rmse == run.data.metrics["test_rmse"]
        assert result.test_r2 == run.data.metrics["test_r2"]
        assert result.best_epoch == 1

    restarted = ExperimentRunner(training=runner.training, tracking=runner.tracker.settings)
    first, second = summary.results
    client = runner.tracker.client
    client.set_tag(first.run_id, "mgs_kan.protocol", "previous-source-version")
    assert restarted.run(**args) == summary
    first_run = client.get_run(first.run_id)
    # Simulate interruption between FileStore's individual initial tag writes.
    client.create_run(first_run.info.experiment_id, tags={
        "mgs_kan.series_key": first_run.data.tags["mgs_kan.series_key"],
    })
    checkpoint = first_run.data.tags["mgs_kan.checkpoint_file"]
    client.set_tag(first.run_id, "mgs_kan.checkpoint_file", "missing-checkpoint.pt")
    assert restarted.run(**args, time_budget_seconds=0).pending == (first.trial,)
    client.set_tag(first.run_id, "mgs_kan.checkpoint_file", checkpoint)
    runner.tracker.client.delete_tag(first.run_id, "mgs_kan.evaluated")
    paused = restarted.run(**args, time_budget_seconds=0)
    assert paused.results == (second,)
    assert paused.pending == (first.trial,)

    # A finished fit with an interrupted evaluation must receive a fresh attempt.
    retried = restarted.run(**args)
    assert retried.results[0].run_id != first.run_id
    assert retried.results[1] == second
    new_seeds = restarted.run(**{**args, "seeds": [(17, 43)]}, time_budget_seconds=0)
    assert not new_seeds.results and len(new_seeds.pending) == 2
    with (args["root"] / "table.csv").open("a") as table:
        table.write("41,82\n")
    assert restarted.run(**args) == retried
    new_version = restarted.run(**{**args, "series_id": "local-series-v2"}, time_budget_seconds=0)
    assert not new_version.results and len(new_version.pending) == 2


def test_duplicate_grid_device_and_budget_rejected_before_probe(local_series, monkeypatch):
    runner, args = local_series
    monkeypatch.setattr(runner, "_worker", lambda _: pytest.fail("unexpected worker"))
    for updates in [dict(folds=[1, 1]), dict(device="cuda"), dict(max_runs=1),
                    dict(time_budget_seconds=-1), dict(seeds=[])]:
        with pytest.raises(ValueError):
            runner.run(**{**args, **updates})


def test_worker_interrupt_is_reaped(local_series, monkeypatch):
    runner, _ = local_series
    runner.resources = ResourceSettings(cpu_threads=3)
    proc = Mock()
    proc.communicate.side_effect = KeyboardInterrupt
    proc.poll.return_value = None
    proc.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), 1]
    popen = Mock(return_value=proc)
    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(KeyboardInterrupt):
        runner._worker({"device": "cpu"})
    proc.send_signal.assert_called_once_with(signal.SIGINT)
    proc.kill.assert_called_once()
    assert proc.wait.call_count == 2
    assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert popen.call_args.kwargs["env"]["MLFLOW_TRACKING_URI"] == runner.tracker.settings.tracking_uri
    assert popen.call_args.kwargs["env"]["OPENBLAS_NUM_THREADS"] == "3"


def test_fold_weighted_summary_reports_partial_coverage():
    a, b, c, d = [Trial("mlp", f, seed, 42) for f in (0, 1) for seed in (1, 2)]
    runs = tuple(TrialResult(t, str(i), 1, rmse, r2)
                 for i, (t, rmse, r2) in enumerate([(a, 2, None), (b, 4, None), (c, 9, 0.5)]))
    summary = SeriesSummary("s", runs, (d,))
    assert summary.by_model == {"mlp": {"completed": 3, "planned": 4,
                                        "test_rmse": 6, "test_r2": 0.5}}


def test_queue_stops_at_failed_seed_pair(local_series, monkeypatch):
    runner, args = local_series
    seen = []

    def simulate(request):
        if request["action"] == "probe":
            return {"identity": {}, "protocols": [], "completed": {}}
        seen.append((request["model"], request["fold"], request["init_seed"], request["shuffle_seed"]))
        if len(seen) == 3:
            raise RuntimeError("CUDA out of memory")
        return {"run_id": str(len(seen))}

    def scores(run_id):
        return dict(run_id=run_id, best_epoch=1, test_rmse=3.0, test_r2=None)

    monkeypatch.setattr(runner.tracker, "trial_scores", scores)
    monkeypatch.setattr(runner, "_worker", simulate)
    with pytest.raises(RuntimeError, match="mlp/1/19/43: CUDA out of memory"):
        runner.run(**{**args, "folds": [1, 2], "seeds": [(17, 42), (19, 43)]})
    assert seen == [("mlp", 1, 17, 42), ("mgs_kan", 1, 17, 42), ("mlp", 1, 19, 43)]


def test_series_order_and_protocol_are_independent_of_runner():
    settings = dict(series_id="demo", dataset="concrete", device="cpu",
                    resources=ResourceSettings(), training=TrainingSettings(), model_settings=ModelSettings())
    series = ExperimentSeries(**settings, models=["mgs_kan", "mlp"], folds=[3, 0], seeds=[(1, 3)])
    reordered = ExperimentSeries(**settings, models=["mlp", "mgs_kan"], folds=[0, 3], seeds=[(1, 3)])
    identity = {"dataset_sha256": "data", "source_sha256": "source", "device_model": "cpu-model"}
    assert [trial.key for trial in series.trials()] == [
        "mgs_kan/3/1/3", "mlp/3/1/3", "mgs_kan/0/1/3", "mlp/0/1/3",
    ]
    assert series.protocol(identity) == reordered.protocol(identity)
    assert series.protocol({**identity, "device_model": "different"}) != series.protocol(identity)


def test_validation_deep_runs_resume_without_test_metrics(local_series):
    runner, args = local_series
    args.update(models=["deep_mlp", "resnet"], evaluation="validation")
    summary = runner.run(**args)
    for result in summary.results:
        run = runner.tracker.client.get_run(result.run_id)
        assert result.best_val_rmse == run.data.metrics["val_rmse"]
        assert result.best_val_rmse == summary.by_model[result.trial.model]["best_val_rmse"]
        assert result.test_rmse is None and result.test_r2 is None
        assert not any(name.startswith("test_") for name in run.data.metrics)
    assert runner.run(**args) == summary
    test_series = runner.run(**{**args, "evaluation": "test"}, time_budget_seconds=0)
    assert not test_series.results and len(test_series.pending) == 2
