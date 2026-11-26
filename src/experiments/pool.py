"""Bounded scheduling of disposable trial processes on one device."""

import subprocess
from concurrent.futures import FIRST_COMPLETED, CancelledError, ThreadPoolExecutor, wait
from threading import Event, Lock
from time import monotonic

from src.experiments.failures import WorkerFailed


def gpu_memory_by_pid(pids: set[int]) -> dict[int, int]:
    """Read driver-reported MiB for our compute processes, including CUDA contexts."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        text=True, timeout=5,
    )
    usage = {}
    for line in output.splitlines():
        pid, _, memory = line.partition(",")
        if pid.strip().isdigit() and int(pid) in pids:
            pid = int(pid)
            usage[pid] = usage.get(pid, 0) + int(memory.strip())
    return usage


class TrialPool:
    def __init__(self, slots: int, budget_mib: int | None):
        self.slots = slots
        self.budget_mib = budget_mib
        self.peak_mib = None
        self._cancel = Event()
        self._lock = Lock()
        self._pids = set()
        self._executor = None

    def __enter__(self):
        if self.slots > 1 or self.budget_mib is not None:
            self._executor = ThreadPoolExecutor(max_workers=self.slots)
        return self

    def __exit__(self, *exc):
        self._cancel.set()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

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
        used = sum(gpu_memory_by_pid(pids).values())
        self.peak_mib = max(self.peak_mib or 0, used)
        if used > self.budget_mib:
            raise WorkerFailed(f"GPU workers use {used} MiB, budget {self.budget_mib} MiB", "cuda_oom", None)

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
