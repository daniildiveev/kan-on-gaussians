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

from config import ModelSettings, ResourceSettings, TrackingSettings, TrainingSettings
from data import CsvSpec
from tracking import MlflowTracker

from .results import SeriesSummary, TrialResult
from .series import ExperimentSeries


class ExperimentRunner:
    """Run one model, fold and seed pair per short-lived worker process."""

    def __init__(
        self, *,
        training: TrainingSettings | None = None,
        resources: ResourceSettings | None = None,
        tracking: TrackingSettings | None = None,
        model_settings: ModelSettings | None = None,
    ):
        self.training = training or TrainingSettings()
        self.resources = resources or ResourceSettings()
        self.tracker = MlflowTracker(tracking)
        self.model_settings = model_settings or ModelSettings()

    def run(
        self, *,
        series_id: str,
        dataset: str | CsvSpec,
        models: list[str],
        folds: list[int],
        seeds: list[tuple[int, int]],
        device: str,
        root: str | Path = "data",
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

        request = {
            **series.model_dump(include={"dataset", "device", "resources", "training", "model_settings"}),
            "root": str(Path(root).resolve()),
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
        print(f"{series_id}: {series.size} planned, {len(completed)} already completed", flush=True)
        for key, trial in trials.items():
            if key in completed:
                continue
            if time_budget_seconds is not None and monotonic() - start >= time_budget_seconds:
                break
            try:
                response = self._worker({
                    **request, **asdict(trial), "expected": identity, "action": "trial",
                    "tags": {**tags, "mgs_kan.trial": key},
                })
                completed[key] = TrialResult(trial, **self.tracker.trial_scores(response["run_id"]))
            except Exception as error:
                raise RuntimeError(f"Stopped series {series_id!r} at {key}: {error}") from error
        return SeriesSummary(
            series_id,
            tuple(completed[k] for k in trials if k in completed),
            tuple(t for k, t in trials.items() if k not in completed),
        )

    def _worker(self, request: dict) -> dict:
        env = os.environ.copy()
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
                [sys.executable, "-m", "experiments.worker", str(result)],
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
                raise RuntimeError(errors.strip() or f"Worker exited with status {proc.returncode}")
            return json.loads(result.read_text(encoding="utf-8"))
