import subprocess
from threading import Barrier, Event
from unittest.mock import Mock

import pytest

from src.experiments.pool import TrialPool, gpu_memory_by_pid


def test_four_slots_work_concurrently_and_return_each_trial():
    first_wave = Barrier(4, timeout=5)

    def execute(trial):
        if trial < 4:
            first_wave.wait()
        return trial

    with TrialPool(slots=4, budget_mib=None) as pool:
        assert sorted(pool.results(range(8), execute, None)) == [(i, i) for i in range(8)]


def test_gpu_sample_sums_only_owned_pids(monkeypatch):
    output = "41, 300\n42, 600\n999, 9000\n"
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: output)
    assert gpu_memory_by_pid({41, 42}) == {41: 300, 42: 600}


def test_budget_breach_cancels_active_work_and_stops_dispatch(monkeypatch):
    from concurrent.futures import CancelledError

    started = []
    active = Event()
    process = Mock(pid=41)

    def communicate(*args, **kwargs):
        active.set()
        Event().wait(0.01)
        raise subprocess.TimeoutExpired("worker", 0.5)

    process.communicate.side_effect = communicate
    monkeypatch.setattr("src.experiments.pool.gpu_memory_by_pid", lambda pids: {41: 4100})

    pool = TrialPool(1, 4096)

    def execute(trial):
        started.append(trial)
        with pytest.raises(CancelledError):
            pool.communicate(process, "payload")

    with pytest.raises(RuntimeError, match="4100 MiB"), pool:
        list(pool.results([1, 2], execute, None))
    assert active.is_set() and started == [1] and pool.peak_mib == 4100
