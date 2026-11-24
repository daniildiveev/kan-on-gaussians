from uuid import uuid4

import pytest
from pydantic import ValidationError


def test_monitor_event_contract():
    from monitoring.events import MonitorEvent

    values = dict(session_id=uuid4(), series_id="demo", sequence=1, kind="started",
                  total=10, device="cuda:1")
    event = MonitorEvent(**values)
    assert event.completed is None
    trial = dict(model="mlp", fold=1, init_seed=2**64 - 1, shuffle_seed=0)
    event = MonitorEvent(**values, trial=trial)
    assert MonitorEvent.model_validate_json(event.model_dump_json()) == event
    for invalid in ({"completed": 11}, {"raw_traceback": "secret"}):
        with pytest.raises(ValidationError):
            MonitorEvent(**values, **invalid)
