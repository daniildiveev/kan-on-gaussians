from threading import Event

import pytest


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
