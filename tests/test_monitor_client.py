from threading import Event

import pytest


def test_grid_reports_each_tenth_once_and_leaves_last_run_to_finish():
    from src.config import MonitorSettings
    from src.experiments.results import TrialResult
    from src.experiments.series import Trial
    from src.monitoring.client import MonitorSession

    settings = MonitorSettings(endpoint="https://monitor.test/v1/events", token="test")
    session = MonitorSession(settings, "stage", 23, "cpu", grid_id="grid")
    result = TrialResult(Trial("mlp", 0, 1, 2), "fresh", 1, None, None, 0.25)
    for completed in range(1, 24):
        session.record_result(completed, result)
    session.record_result(23, result)
    event = session._snapshot("heartbeat")
    assert [(p.percent, p.completed) for p in event.milestones] == [
        (10, 3), (20, 5), (30, 7), (40, 10), (50, 12),
        (60, 14), (70, 17), (80, 19), (90, 21),
    ]


def test_first_fresh_result_wakes_grid_monitor_before_next_heartbeat(monkeypatch):
    from src.config import MonitorSettings
    from src.experiments.results import TrialResult
    from src.experiments.series import Trial
    from src.monitoring.client import MonitorSession

    sent = []
    delivered = Event()

    def post(session, event):
        sent.append(event)
        if event.kind == "heartbeat" and event.first_result:
            delivered.set()
        return 204

    monkeypatch.setattr(MonitorSession, "_post", post)
    settings = MonitorSettings(endpoint="https://monitor.test/v1/events", token="test")
    session = MonitorSession(settings, "grid", 20, "cpu", grid_id="grid")
    session.start()
    try:
        session.begin_series("stage", 12)
        result = TrialResult(Trial("deep_mlp", 0, 1, 2), "fresh", 1, None, None, 0.25)
        session.record_result(13, result)
        assert delivered.wait(2)
        assert sent[-1].first_result.run_id == "fresh" and sent[-1].milestones is None
    finally:
        session.finish("interrupted")


def test_heartbeat_while_runner_is_blocked_and_final_retry(monkeypatch, tmp_path):
    from src.config import MonitorSettings
    from src.monitoring.client import MonitorSession

    received = Event()
    sent = []

    def post(self, event):
        sent.append(event)
        if event.kind == "heartbeat":
            received.set()
        if event.kind == "finished" and sent[-2].kind != "finished":
            raise OSError("offline")
        return 204

    monkeypatch.setattr(MonitorSession, "_post", post)
    monkeypatch.setattr(MonitorSession, "interval", 0.02)
    settings = MonitorSettings(endpoint="https://example.test/v1/events", token="example",
                               disk_path=tmp_path, disk_threshold_gib=1)
    session = MonitorSession(settings, "demo", 2, "cpu")
    session.start()
    try:
        session.update(completed=1)
        assert received.wait(2)
    finally:
        session.finish("finished")
    assert sent[0].kind == "started"
    assert sent[-1] == sent[-2]
    assert sent[-1].completed == 1
    assert sent[-1].disk.path == str(tmp_path)


def test_retry_registration_and_disabled_mode(monkeypatch, caplog):
    from src.config import MonitorSettings
    from src.monitoring.client import MonitorSession

    sent = []

    def post(self, event):
        sent.append(event)
        if len(sent) == 1:
            raise OSError("private error")
        return 204

    monkeypatch.setattr(MonitorSession, "_post", post)
    session = MonitorSession(MonitorSettings(), "test", 1, "cpu")
    session.start()
    session.finish("finished")
    assert sent == []
    session = MonitorSession(MonitorSettings(endpoint="https://monitor.test/v1/events",
                             token="private-token"), "test", 1, "cpu")
    session.start()
    session.finish("paused")
    assert [e.kind for e in sent] == ["started", "started", "paused"]
    assert sent[0] == sent[1]
    assert "monitor" in caplog.text.lower()
    assert "private" not in caplog.text


def test_endpoint_requires_https_and_secret():
    from src.config import MonitorSettings
    from pydantic import ValidationError

    for url in ("http://local/v1/events", "https://name:password@local/v1/events"):
        with pytest.raises(ValidationError):
            MonitorSettings(endpoint=url, token="abc")
    with pytest.raises(ValidationError):
        MonitorSettings(endpoint="https://monitor.test/v1/events")


def test_stalled_transport_cannot_hold_up_training_or_shutdown(monkeypatch):
    from threading import Thread
    from src.config import MonitorSettings
    from src.monitoring.client import MonitorSession

    release = Event()
    entered = Event()
    finished = Event()

    def stalled(self, event):
        entered.set()
        release.wait(5)
        return 204

    monkeypatch.setattr(MonitorSession, "_post", stalled)
    monkeypatch.setattr(MonitorSession, "io_timeout", 0.05, raising=False)
    settings = MonitorSettings(endpoint="https://monitor.test/v1/events", token="example")
    session = MonitorSession(settings, "demo", 1, "cpu")

    def run():
        session.start()
        session.finish("failed")
        finished.set()

    thread = Thread(target=run)
    thread.start()
    try:
        assert entered.wait(1)
        assert finished.wait(1)
    finally:
        release.set()
        thread.join(2)


def test_hypergraph_grid_reports_milestones_supported_by_existing_monitor():
    from src.config import MonitorSettings
    from src.experiments.results import TrialResult
    from src.experiments.series import Trial
    from src.monitoring.client import MonitorSession

    settings = MonitorSettings(endpoint="https://monitor.test/v1/events", token="test")
    session = MonitorSession(settings, "gas", 18, "cpu", grid_id="hypergraph")
    result = TrialResult(Trial("hypergraph_mgs_kan", 0, 11, 101), "fresh", 1, None, None, 0.25)
    session.record_result(13, result)
    event = session._snapshot("heartbeat")
    assert event.completed == 13 and event.first_result is None
    assert event.milestones[-1].percent == 70
