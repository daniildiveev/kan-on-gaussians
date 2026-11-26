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


def test_hidden_pids_monitor_one_card_for_four_slots(monkeypatch):
    uuid = "GPU-0c5969a5-8446-d137-5d5c-aee5273f4349"
    samples = iter(["3072", "4101"])

    def query(command, **kwargs):
        assert f"--id={uuid}" in command
        assert "--query-gpu=memory.used" in command
        return next(samples)

    monkeypatch.setattr(subprocess, "check_output", query)
    with TrialPool(4, 4096) as pool:
        pool._pids.update([433131, 433132, 433133, 433134])
        (pool.fallback_dir / uuid).touch()
        pool._sample_memory()
        assert pool.peak_device_mib == 3072 and pool.peak_mib is None
        with pytest.raises(RuntimeError, match="4101 MiB"):
            pool._sample_memory()


def test_fallback_resolves_visible_cuda_device_uuid(monkeypatch):
    from types import SimpleNamespace
    from uuid import UUID

    import torch
    from src.experiments.gpu_memory import selected_gpu_uuid

    uuid = UUID("0c5969a5-8446-d137-5d5c-aee5273f4349")
    device = torch.device("cuda:1")

    def properties(selected):
        assert selected == device
        return SimpleNamespace(uuid=uuid.bytes)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    assert selected_gpu_uuid(device) == f"GPU-{uuid}"


def test_unavailable_device_reading_is_not_zero(monkeypatch):
    from src.experiments.gpu_memory import device_memory_mib

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "[N/A]")
    with pytest.raises(RuntimeError, match="no usable memory"):
        device_memory_mib("GPU-0c5969a5-8446-d137-5d5c-aee5273f4349")


def test_uuid_from_cuda_driver_when_torch_omits_property(monkeypatch):
    import ctypes
    import os
    from uuid import UUID

    import torch
    from src.experiments.gpu_memory import selected_gpu_uuid

    uuid = UUID("0c5969a5-8446-d137-5d5c-aee5273f4349")

    class Driver:
        def cuDeviceGet(self, address, index):
            assert index == 1
            ctypes.cast(address, ctypes.POINTER(ctypes.c_int))[0] = 7
            return 0

        def cuDeviceGetUuid(self, address, device):
            assert device.value == 7
            ctypes.memmove(address, uuid.bytes, 16)
            return 0

    loader = "WinDLL" if os.name == "nt" else "CDLL"
    monkeypatch.setattr(ctypes, loader, lambda name: Driver())
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: object())
    assert selected_gpu_uuid(torch.device("cuda:1")) == f"GPU-{uuid}"
