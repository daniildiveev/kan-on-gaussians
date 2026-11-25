import pytest

pytest.importorskip("mlflow")


def test_real_series_sends_progress_and_resume(tmp_path, monkeypatch, run_log_dir):
    from src.config import MonitorSettings, TrackingSettings, TrainingSettings
    from src.data import CsvSpec
    from src.experiments import ExperimentRunner
    from src.monitoring.client import MonitorSession

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    path = tmp_path / "table.csv"
    path.write_text("x,y\n" + "".join(f"{i},{i*i}\n" for i in range(40)))
    events = []

    def post(self, event):
        events.append(event)
        return 204

    monkeypatch.setattr(MonitorSession, "_post", post)
    runner = ExperimentRunner(
        training=TrainingSettings(epochs=1),
        tracking=TrackingSettings(tracking_uri=(tmp_path / "runs").as_uri()),
        monitoring=MonitorSettings(endpoint="https://monitor.test/v1/events", token="secret"),
    )
    args = dict(series_id="demo", models=["mlp"], dataset=CsvSpec("table.csv", "y"),
                seeds=[(17, 42)], folds=[0], root=tmp_path, device="cpu")
    partial = runner.run(**args, time_budget_seconds=0)
    assert partial.pending and events[-1].kind == "paused"
    first = runner.run(**args)
    assert runner.run(**args) == first
    logs = [path.read_text() for path in run_log_dir.glob("series-*.log")]
    assert len(logs) == 3
    assert any("Series paused: completed=0/1" in log for log in logs)
    assert any(f"run_id={first.results[0].run_id} val_rmse=" in log and "val_rmse=None" not in log for log in logs)
    assert any("reused=1 planned=1" in log and "Training started:" not in log for log in logs)
    assert events[-1].kind == "finished" and events[-1].completed == 1
    assert len({e.session_id for e in events}) == 3
    assert all(e.completed is None for e in events if e.kind == "started")

    def broken(request):
        raise RuntimeError("probe failure")

    monkeypatch.setattr(runner, "_worker", broken)
    with pytest.raises(RuntimeError, match="probe failure"):
        runner.run(**args)
    assert events[-1].kind == "failed" and events[-1].completed is None
    assert events[-1].run_id is None

    def interrupt(request):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_worker", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run(**args)
    assert events[-1].kind == "interrupted"


def test_worker_failure_code_from_wrapped_mlflow_timeout():
    from mlflow.exceptions import MlflowException
    from requests.exceptions import Timeout
    from src.experiments.failures import failure_reason

    try:
        raise Timeout()
    except Timeout as timeout:
        error = MlflowException("tracking request failed")
        error.__cause__ = timeout
    assert failure_reason(error) == "mlflow_unavailable"
    assert failure_reason(Timeout()) == "series_error"


def test_interrupt_after_registration_closes_session(tmp_path, monkeypatch):
    from src.config import MonitorSettings, TrackingSettings
    from src.experiments import ExperimentRunner
    from src.monitoring.client import MonitorSession

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    received = []

    def post(self, event):
        received.append(event.kind)
        return 204

    start = MonitorSession.start

    def interrupt(session):
        start(session)
        raise KeyboardInterrupt

    monkeypatch.setattr(MonitorSession, "_post", post)
    monkeypatch.setattr(MonitorSession, "start", interrupt)
    runner = ExperimentRunner(
        tracking=TrackingSettings(tracking_uri=(tmp_path / "runs").as_uri()),
        monitoring=MonitorSettings(endpoint="https://monitor.test/v1/events", token="test"),
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run(series_id="demo", dataset="concrete", folds=[0],
                   seeds=[(1, 2)], models=["mlp"], device="cpu")
    assert received == ["started", "interrupted"]


def test_worker_secrets_and_failed_trial_context(tmp_path, monkeypatch):
    import json
    import subprocess
    from pathlib import Path
    from unittest.mock import Mock

    from src.config import TrackingSettings
    from src.experiments import ExperimentRunner
    from src.monitoring.client import MonitorSession

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MGS_KAN_MONITOR_TOKEN", "worker-must-not-see-this")
    monkeypatch.setenv("MGS_KAN_MONITOR_ENDPOINT", "https://example.test/v1/events")
    events = []

    def post(self, event):
        events.append(event)
        return 204

    def launch(args, **kwargs):
        assert "MGS_KAN_MONITOR_TOKEN" not in kwargs["env"]
        process = Mock()

        def communicate(payload):
            assert "worker-must-not-see-this" not in payload
            is_probe = json.loads(payload)["action"] == "probe"
            process.returncode = 0 if is_probe else 1
            response = ({"identity": {}, "protocols": [], "completed": {}} if is_probe
                        else {"run_id": "known", "reason": "cuda_oom"})
            Path(args[-1]).write_text(json.dumps(response))
            return "", "original traceback"

        process.communicate.side_effect = communicate
        return process

    monkeypatch.setattr(MonitorSession, "_post", post)
    runner = ExperimentRunner(tracking=TrackingSettings(tracking_uri=(tmp_path / "runs").as_uri()))
    monkeypatch.setattr(subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="original traceback"):
        runner.run(series_id="demo", dataset="concrete", folds=[0],
                   seeds=[(1, 2)], models=["mlp"], device="cpu")
    assert events[-1].reason == "cuda_oom" and events[-1].run_id == "known"
    assert events[-1].trial.shuffle_seed == 2 and events[-1].completed == 0
