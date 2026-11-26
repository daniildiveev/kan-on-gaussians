"""Schedule a series and manage its disposable worker processes."""

import json
import logging
import math
import os
import signal
import subprocess
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Literal

from src.config import ModelSettings, MonitorSettings, ResourceSettings, TrackingSettings, TrainingSettings
from src.data import CsvSpec
from src.monitoring.client import MonitorSession
from src.monitoring.events import TrialSnapshot
from src.paths import DATA_DIR, PROJECT_ROOT, project_path
from src.tracking import MlflowTracker

from src.experiments.failures import WorkerFailed, failure_reason
from src.experiments.pool import TrialPool
from src.experiments.results import SeriesSummary, TrialResult
from src.experiments.run_log import run_log
from src.experiments.series import ExperimentSeries, Trial


class TrialFailed(RuntimeError):
    def __init__(self, series_id, trial, error):
        super().__init__(f"Stopped series {series_id!r} at {trial.key}: {error}")
        self.trial = trial
        self.run_id = error.run_id if isinstance(error, WorkerFailed) else None


def _snapshot(trial: Trial):
    # Older monitor deployments accept only the two compact model names.
    return TrialSnapshot(**asdict(trial)) if trial.model in ("mlp", "mgs_kan") else None


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
        feature_columns: tuple[str, ...] | None = None,
        evaluation: Literal["test", "validation"] = "test",
        max_runs: int | None = None,
        time_budget_seconds: float | None = None,
    ) -> SeriesSummary:
        start = monotonic()
        series = ExperimentSeries(
            series_id=series_id, dataset=dataset, models=models, folds=folds, seeds=seeds,
            device=device, training=self.training, resources=self.resources,
            model_settings=self.model_settings, evaluation=evaluation, feature_columns=feature_columns,
        )
        if max_runs is not None and (type(max_runs) is not int or max_runs < series.size):
            raise ValueError(f"The full grid has {series.size} runs and exceeds max_runs={max_runs}")
        if time_budget_seconds is not None and not (0 <= time_budget_seconds < math.inf):
            raise ValueError("time_budget_seconds must be finite and nonnegative")

        session = MonitorSession(self.monitoring, series_id, series.size, device)
        try:
            session.start()
            with run_log("series") as log:
                log.info("Series started: series=%s dataset=%s device=%s planned=%d",
                         series_id, dataset, device, series.size)
                deadline = None if time_budget_seconds is None else start + time_budget_seconds
                summary = self.run_series(series, session, root=root, download=download, deadline=deadline, log=log)
                status = "paused" if summary.pending else "finished"
                log.info("Series %s: completed=%d/%d elapsed=%.2fs",
                         status, len(summary.results), series.size, monotonic() - start)
        except KeyboardInterrupt:
            session.finish("interrupted")
            raise
        except Exception as error:
            session.finish("failed", failure_reason(error))
            raise
        session.finish(status)
        return summary

    def run_series(
        self, series: ExperimentSeries, session: MonitorSession, *, root: str | Path,
        download: bool, completed_before: int = 0, deadline: float | None = None,
        log: logging.Logger = logging.getLogger(__name__),
    ) -> SeriesSummary:
        """Execute a series inside the caller's monitoring session."""
        series_id = series.series_id
        log.info("Checking MLflow history: series=%s", series_id)
        request = {
            **series.model_dump(include={"dataset", "device", "resources", "training", "model_settings", "feature_columns"}),
            "root": str(project_path(root)),
            "download": download,
            "evaluation": series.evaluation,
            "tracking": self.tracker.settings.model_dump(),
        }
        probe = self._worker({
            **request, "action": "probe", "fold": series.folds[0], "series_key": series.key,
        })
        identity = probe["identity"]
        tags = {
            "mgs_kan.series": series_id,
            "mgs_kan.series_key": series.key,
            "mgs_kan.protocol": series.protocol(identity),
            "mgs_kan.evaluation": series.evaluation,
        }
        trials = {trial.key: trial for trial in series.trials()}
        completed = {
            key: TrialResult(trials[key], **values)
            for key, values in probe["completed"].items() if key in trials
        }
        session.update(completed=completed_before + len(completed))
        log.info("Series resume: series=%s reused=%d planned=%d", series_id, len(completed), series.size)
        print(f"{series_id}: {series.size} planned, {len(completed)} already completed", flush=True)
        resources = series.resources
        budget = resources.total_gpu_budget_mib if series.device.startswith("cuda:") else None
        pending = [trial for key, trial in trials.items() if key not in completed]
        reused = len(completed)
        request.update(action="trial", expected=identity, tags=tags)
        pool = TrialPool(resources.max_parallel_trials, budget, probe.get("whole_gpu_uuid"))
        execute = partial(self._execute_trial, request=request, pool=pool, session=session, log=log)
        try:
            with pool:
                for trial, (result, details) in pool.results(pending, execute, deadline):
                    completed[trial.key] = result
                    count = completed_before + len(completed)
                    log.info("Training finished: %s completed=%d/%d", details, count, session.identity["total"])
                    session.record_result(count, result)
        except TrialFailed as error:
            session.update(trial=_snapshot(error.trial), run_id=error.run_id)
            raise
        if pool.peak_device_mib is not None:
            log.info("Sampled peak for entire GPU %s: %d MiB", pool.whole_gpu_uuid, pool.peak_device_mib)
        if pool.peak_mib is not None:
            log.info("Sampled peak GPU memory for training workers: %d MiB", pool.peak_mib)
        return SeriesSummary(
            series_id,
            tuple(completed[k] for k in trials if k in completed),
            tuple(t for k, t in trials.items() if k not in completed),
            series.evaluation, reused=reused, peak_gpu_mib=pool.peak_mib, peak_device_mib=pool.peak_device_mib,
        )

    def _execute_trial(self, trial: Trial, *, request: dict, pool: TrialPool, session: MonitorSession,
                       log: logging.Logger):
        tags = request["tags"]
        series_id = tags["mgs_kan.series"]
        context = (f"series={series_id} model={trial.model} fold={trial.fold} "
                   f"init_seed={trial.init_seed} shuffle_seed={trial.shuffle_seed}")
        if pool.slots == 1:
            session.update(trial=_snapshot(trial))
        started = monotonic()
        log.info("Training started: %s", context)
        job = {**request, **asdict(trial), "tags": {**tags, "mgs_kan.trial": trial.key}}
        if pool.whole_gpu_uuid is not None:
            job["whole_gpu_uuid"] = pool.whole_gpu_uuid
        elif pool.fallback_dir is not None:
            job["fallback_dir"] = str(pool.fallback_dir)
        try:
            if pool.slots == 1 and pool.budget_mib is None:
                response = self._worker(job)
            else:
                response = self._worker(job, pool=pool)
            scores = self.tracker.trial_scores(response["run_id"])
        except Exception as error:
            raise TrialFailed(series_id, trial, error) from error
        result = TrialResult(trial, **scores)
        val_rmse = response.get("val_rmse", result.best_val_rmse)
        details = f"{context} run_id={result.run_id} val_rmse={val_rmse} elapsed={monotonic() - started:.2f}s"
        return result, details

    def _worker(self, request: dict, *, pool: TrialPool | None = None) -> dict:
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
                message = json.dumps(payload) + "\n"
                if pool is None:
                    _, errors = proc.communicate(message)
                else:
                    _, errors = pool.communicate(proc, message)
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
