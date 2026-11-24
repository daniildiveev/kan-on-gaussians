"""Schedule a series and manage its disposable worker processes."""

import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from src.config import ModelSettings, MonitorSettings, ResourceSettings, TrackingSettings, TrainingSettings
from src.data import CsvSpec
from src.monitoring.client import MonitorSession
from src.monitoring.events import TrialSnapshot
from src.paths import DATA_DIR, PROJECT_ROOT, project_path
from src.tracking import MlflowTracker

from src.experiments.failures import WorkerFailed, failure_reason
from src.experiments.results import SeriesSummary, TrialResult
from src.experiments.series import ExperimentSeries


class ExperimentRunner:
    """Run one model, fold and seed pair per short-lived worker process."""

    def __init__(
        self, *,
        training: TrainingSettings | None = None,
        resources: ResourceSettings | None = None,
        tracking: TrackingSettings | None = None,
        model_settings: ModelSettings | None = None,
        monitoring: MonitorSettings | None = None,
    ):
        self.training = training or TrainingSettings()
        self.resources = resources or ResourceSettings()
        self.tracker = MlflowTracker(tracking)
        self.model_settings = model_settings or ModelSettings()
        self.monitoring = monitoring or MonitorSettings()

    def run(
        self, *,
        series_id: str,
        dataset: str | CsvSpec,
        models: list[str],
        folds: list[int],
        seeds: list[tuple[int, int]],
        device: str,
        root: str | Path = DATA_DIR,
        download: bool = False,
        max_runs: int | None = None,
        time_budget_seconds: float | None = None,
    ) -> SeriesSummary:
        start = monotonic()
        series = ExperimentSeries(
            series_id=series_id, dataset=dataset, models=models, folds=folds, seeds=seeds,
            device=device, training=self.training, resources=self.resources,
            model_settings=self.model_settings,
        )
        if max_runs is not None and (type(max_runs) is not int or max_runs < series.size):
            raise ValueError(f"The full grid has {series.size} runs and exceeds max_runs={max_runs}")
        if time_budget_seconds is not None and not (0 <= time_budget_seconds < math.inf):
            raise ValueError("time_budget_seconds must be finite and nonnegative")

        session = MonitorSession(self.monitoring, series_id, series.size, device)
        try:
            session.start()
            summary = self._run_series(series, session, root, download, start, time_budget_seconds)
        except KeyboardInterrupt:
            session.finish("interrupted")
            raise
        except Exception as error:
            session.finish("failed", failure_reason(error))
            raise
        session.finish("paused" if summary.pending else "finished")
        return summary

    def _run_series(
        self, series: ExperimentSeries, session: MonitorSession, root: str | Path,
        download: bool, start: float, time_budget_seconds: float | None,
    ) -> SeriesSummary:
        series_id = series.series_id
        request = {
            **series.model_dump(include={"dataset", "device", "resources", "training", "model_settings"}),
            "root": str(project_path(root)),
            "download": download,
            "tracking": self.tracker.settings.model_dump(),
        }
        probe = self._worker({
            **request, "action": "probe", "fold": series.folds[0], "series_key": series.key,
        })
        identity = probe["identity"]
        protocol = series.protocol(identity)
        if any(saved != protocol for saved in probe["protocols"]):
            raise ValueError("Series protocol changed. Use a new series_id.")
        tags = {
            "mgs_kan.series": series_id,
            "mgs_kan.series_key": series.key,
            "mgs_kan.protocol": protocol,
        }
        trials = {trial.key: trial for trial in series.trials()}
        completed = {
            key: TrialResult(trials[key], **values) for key, values in probe["completed"].items()
        }
        session.update(completed=len(completed))
        print(f"{series_id}: {series.size} planned, {len(completed)} already completed", flush=True)
        for key, trial in trials.items():
            if key in completed:
                continue
            if time_budget_seconds is not None and monotonic() - start >= time_budget_seconds:
                break
            snapshot = TrialSnapshot(**asdict(trial))
            session.update(trial=snapshot)
            try:
                response = self._worker({
                    **request, **asdict(trial), "expected": identity, "action": "trial",
                    "tags": {**tags, "mgs_kan.trial": key},
                })
                run_id = response["run_id"]
                session.update(trial=snapshot, run_id=run_id)
                completed[key] = TrialResult(trial, **self.tracker.trial_scores(run_id))
                session.update(completed=len(completed))
            except Exception as error:
                if isinstance(error, WorkerFailed):
                    session.update(trial=snapshot, run_id=error.run_id)
                raise RuntimeError(f"Stopped series {series_id!r} at {key}: {error}") from error
        return SeriesSummary(
            series_id,
            tuple(completed[k] for k in trials if k in completed),
            tuple(t for k, t in trials.items() if k not in completed),
        )

    def _worker(self, request: dict) -> dict:
        env = os.environ.copy()
        # A notebook can import a checkout that is not installed in the kernel environment.
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + pythonpath if pythonpath else "")
        for name in list(env):
            if name.upper().startswith("MGS_KAN_MONITOR_"):
                del env[name]
        env["MLFLOW_ENABLE_ASYNC_LOGGING"] = "false"
        env["MLFLOW_TRACKING_URI"] = self.tracker.settings.tracking_uri
        if request["device"] == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(self.resources.cpu_threads)
        with TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            proc = subprocess.Popen(
                [sys.executable, "-P", "-m", "src.experiments.worker", str(result)],
                stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                start_new_session=True,
            )
            try:
                payload = {**request, "parent_pid": os.getpid()}
                _, errors = proc.communicate(json.dumps(payload) + "\n")
            except BaseException:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                raise
            if proc.returncode:
                details = {}
                if result.is_file():
                    try:
                        details = json.loads(result.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        pass  # A killed worker may have only partially written its result.
                raise WorkerFailed(errors.strip() or f"Worker exited with status {proc.returncode}",
                                   details.get("reason", "worker_exit"), details.get("run_id"))
            return json.loads(result.read_text(encoding="utf-8"))
