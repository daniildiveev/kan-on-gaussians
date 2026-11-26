"""Best-effort, ordered heartbeats from the parent notebook process."""

import logging
import math
import shutil
from contextlib import closing
from http.client import HTTPSConnection
from threading import Event, Lock, Thread
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4

from src.config import MonitorSettings
from src.monitoring.events import DiskSample, FirstResult, GridMilestone, Kind, MonitorEvent, Reason, TrialSnapshot

if TYPE_CHECKING:
    from src.experiments.results import TrialResult


class MonitorSession:
    interval = 60
    io_timeout = 3

    def __init__(self, settings: MonitorSettings, series_id: str, total: int, device: str,
                 *, grid_id: str | None = None, started_at: float | None = None):
        self.settings = settings
        self.identity = dict(session_id=uuid4(), series_id=series_id[:160], total=total,
                             device=device[:64], grid_id=grid_id[:160] if grid_id else None)
        self._progress = {"completed": None, "trial": None, "run_id": None}
        self._lock = Lock()
        self._started_at = monotonic() if started_at is None else started_at
        self._wake = Event()
        self._stop = Event()
        self._first_attempt = Event()
        self._final_kind: Kind = "interrupted"
        self._final_reason: Reason | None = None
        self._thread = None
        self._sequence = 0
        self._started = None
        self._registered = False
        self._offline = False

    def begin_series(self, series_id: str, completed_before: int) -> None:
        with self._lock:
            self.identity["series_id"] = series_id[:160]
            self._progress.update(completed=completed_before, trial=None, run_id=None)

    def update(self, completed: int | None = None, trial: TrialSnapshot | None = None,
               run_id: str | None = None) -> None:
        with self._lock:
            if completed is not None:
                self._progress["completed"] = completed
            self._progress.update(trial=trial, run_id=run_id)

    def record_result(self, completed: int, result: "TrialResult") -> None:
        """Capture only newly trained results. Keep notices until the terminal snapshot."""
        with self._lock:
            self._progress.update(completed=completed, trial=None, run_id=None)
            if self.identity["grid_id"] is None or self.settings.endpoint is None:
                return
            score = result.best_val_rmse
            if "first_result" not in self._progress and score is not None and 0 <= score < math.inf:
                self._progress["first_result"] = FirstResult(
                    model=result.trial.model, fold=result.trial.fold, run_id=result.run_id, val_rmse=score,
                )
                self._wake.set()
            total = self.identity["total"]
            step = completed * 10 // total
            previous_step = (completed - 1) * 10 // total
            if completed == total or step == previous_step:
                return
            percent = 10 * step
            milestones = self._progress.get("milestones", ())
            if any(point.percent == percent for point in milestones):
                return
            point = GridMilestone(
                percent=percent, completed=completed, series_id=self.identity["series_id"],
                elapsed_seconds=int(monotonic() - self._started_at),
            )
            self._progress["milestones"] = (*milestones, point)

    def _snapshot(self, kind: Kind, reason: Reason | None = None) -> MonitorEvent:
        disk = None
        path = self.settings.disk_path
        if path is not None:
            try:
                free_gib = shutil.disk_usage(path).free / 2**30
            except OSError:
                pass
            else:
                disk = DiskSample(path=str(path)[:512], free_gib=free_gib,
                                  threshold_gib=self.settings.disk_threshold_gib,
                                  margin_gib=self.settings.disk_margin_gib)
        self._sequence += 1
        with self._lock:
            return MonitorEvent(**self.identity, **self._progress, sequence=self._sequence,
                                kind=kind, disk=disk, reason=reason)

    def _post(self, event: MonitorEvent) -> int:
        url = self.settings.endpoint
        headers = {"Authorization": "Bearer " + self.settings.token.get_secret_value(),
                   "Content-Type": "application/json"}
        # Direct TLS request. http.client never follows redirects with our token.
        with closing(HTTPSConnection(url.host.strip("[]"), port=url.port, timeout=self.io_timeout)) as conn:
            conn.request("POST", url.path or "/", event.model_dump_json(exclude_none=True).encode(), headers)
            return conn.getresponse().status

    def _send(self, event: MonitorEvent) -> bool:
        try:
            status = self._post(event)
        except Exception:
            # Monitoring is best effort. Network exceptions may contain credentials.
            status = None
        if status is not None and 200 <= status < 300:
            self._offline = False
            return True
        if status == 409:
            self._registered = False
        if not self._offline:
            logging.warning("Monitor delivery failed (%s); training continues", status or "connection")
        self._offline = True
        return False

    def _register(self) -> bool:
        if not self._registered:
            self._registered = self._send(self._started)
        return self._registered

    def start(self) -> None:
        if self.settings.endpoint is None:
            return
        self._started = self._snapshot("started")
        self._thread = Thread(target=self._deliver, name="series-heartbeat", daemon=True)
        self._thread.start()
        if not self._first_attempt.wait(self.io_timeout):
            logging.warning("Monitor registration still pending; training continues")

    def _deliver(self) -> None:
        self._register()
        self._first_attempt.set()
        while True:
            self._wake.wait(self.interval)
            self._wake.clear()
            if self._stop.is_set():
                break
            if self._register():
                self._send(self._snapshot("heartbeat"))
        final = self._snapshot(self._final_kind, self._final_reason)
        for _ in range(2):
            if self._register() and self._send(final):
                return
        logging.warning("Monitor did not confirm the end of this session")

    def finish(self, kind: Kind, reason: Reason | None = None, *, elapsed_seconds: int | None = None) -> None:
        if self._thread is None:
            return
        self._final_kind = kind
        self._final_reason = reason
        with self._lock:
            self._progress["elapsed_seconds"] = elapsed_seconds
        self._stop.set()
        self._wake.set()
        # Bound the caller's wait even if OS-level DNS resolution stalls.
        self._thread.join(2 * self.io_timeout)
        if self._thread.is_alive():
            logging.warning("Monitor final delivery still pending; releasing runner")
