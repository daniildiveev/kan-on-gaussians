"""Bounded scheduling of disposable trial processes on one device."""

import subprocess
from concurrent.futures import FIRST_COMPLETED, CancelledError, ThreadPoolExecutor, wait
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from time import monotonic

from src.experiments.failures import WorkerFailed
from src.experiments.gpu_memory import device_memory_mib, gpu_memory_by_pid


class TrialPool:
    def __init__(self, slots: int, budget_mib: int | None, whole_gpu_uuid: str | None = None):
        self.slots = slots
        self.budget_mib = budget_mib
        self.peak_mib = None
        self.peak_device_mib = None
        self.whole_gpu_uuid = whole_gpu_uuid
        self._directory = None
        self.fallback_dir = None
        self._cancel = Event()
        self._lock = Lock()
        self._pids = set()
        self._executor = None

    def __enter__(self):
        if self.budget_mib is not None and self.whole_gpu_uuid is None:
            self._directory = TemporaryDirectory()
            self.fallback_dir = Path(self._directory.name)
        if self.slots > 1 or self.budget_mib is not None:
            self._executor = ThreadPoolExecutor(max_workers=self.slots)
        return self

    def __exit__(self, *exc):
        self._cancel.set()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        if self._directory is not None:
            self._directory.cleanup()

    def communicate(self, process, payload):
        """Let a worker thread reap its own child when the coordinator stops."""
        with self._lock:
            self._pids.add(process.pid)
        try:
            while not self._cancel.is_set():
                try:
                    return process.communicate(payload, timeout=0.5)
                except subprocess.TimeoutExpired:
                    payload = None  # communicate retains partially written input.
            raise CancelledError("Series stopped")
        finally:
            with self._lock:
                self._pids.discard(process.pid)

    def _sample_memory(self):
        with self._lock:
            pids = self._pids.copy()
        if self.budget_mib is None or not pids:
            return
        if self.whole_gpu_uuid is None and self.fallback_dir is not None:
            # A trial may lack PID visibility even if the probe saw its own PID.
            marker = next(self.fallback_dir.iterdir(), None)
            if marker is not None:
                self.whole_gpu_uuid = marker.name
        if self.whole_gpu_uuid is not None:
            used = device_memory_mib(self.whole_gpu_uuid)
            self.peak_device_mib = max(self.peak_device_mib or 0, used)
            scope = f"GPU {self.whole_gpu_uuid}"
        else:
            readings = gpu_memory_by_pid(pids)
            if not readings:
                return
            used = sum(readings.values())
            self.peak_mib = max(self.peak_mib or 0, used)
            scope = "GPU workers"
        if used > self.budget_mib:
            raise WorkerFailed(f"{scope}: {used} MiB, budget {self.budget_mib} MiB", "cuda_oom", None)

    def results(self, trials, execute, deadline):
        """Yield finished attempts on the coordinator thread. Bound the in-flight queue."""
        if self._executor is None:
            for trial in trials:
                if deadline is not None and monotonic() >= deadline:
                    break
                yield trial, execute(trial)
            return

        pending = {}
        trials = iter(trials)
        next_sample = 0
        while True:
            while len(pending) < self.slots and (deadline is None or monotonic() < deadline):
                trial = next(trials, None)
                if trial is None:
                    break
                pending[self._executor.submit(execute, trial)] = trial
            if not pending:
                return
            done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            if monotonic() >= next_sample:
                self._sample_memory()
                next_sample = monotonic() + 0.5
            for future in done:
                yield pending.pop(future), future.result()
