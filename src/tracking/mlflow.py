"""Record training runs on an existing MLflow tracking server."""

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from typing import TYPE_CHECKING

import torch
from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from torch import nn

from src.config import DEFAULT_CHECKPOINT_FILE, ModelSettings, TrackingSettings, TrainingSettings
from src.data import DatasetSplits
from src.models import parameter_count

from src.tracking.provenance import SourceVersion
from src.tracking.recipe import ValidationRecipe, optimizer_params

if TYPE_CHECKING:
    from src.experiments.series import ExperimentSeries


_CHECKPOINT_TAG = "mgs_kan.checkpoint_file"


@dataclass(frozen=True)
class MlflowRun:
    client: MlflowClient
    run_id: str
    checkpoint_file: str

    def log_metrics(self, step: int, **values: float) -> None:
        timestamp = int(time() * 1000)
        metrics = [Metric(key, value, timestamp, step) for key, value in values.items()]
        self.client.log_batch(self.run_id, metrics=metrics)

    def upload_checkpoint(self, content: bytes) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / self.checkpoint_file
            path.write_bytes(content)
            self.client.log_artifact(self.run_id, str(path))


@dataclass(frozen=True)
class MatchingTrials:
    dataset_sha256: str
    completed: dict[str, dict]


class MlflowTracker:
    def __init__(self, settings: TrackingSettings | None = None, *, run_tags: dict[str, str] | None = None):
        self.settings = settings or TrackingSettings()
        self.client = MlflowClient(tracking_uri=self.settings.tracking_uri)
        self.run_tags = dict(run_tags or {})
        self.last_run_id: str | None = None

    def download_checkpoint(self, run_id: str) -> bytes:
        tags = self.client.get_run(run_id).data.tags
        filename = tags.get(_CHECKPOINT_TAG, DEFAULT_CHECKPOINT_FILE)
        TrackingSettings.validate_checkpoint_file(filename)
        with TemporaryDirectory() as directory:
            path = self.client.download_artifacts(run_id, filename, directory)
            return Path(path).read_bytes()

    def trial_scores(self, run_id: str) -> dict:
        return self._scores(self.client.get_run(run_id))

    @staticmethod
    def _scores(run) -> dict:
        metrics = run.data.metrics
        return {
            "run_id": run.info.run_id,
            "best_epoch": int(metrics["best_epoch"]),
            "test_rmse": metrics.get("test_rmse"),
            "test_r2": metrics.get("test_r2"),
            "best_val_rmse": metrics.get("best_val_rmse"),
        }

    def completed_trials(self, series_key: str, evaluation: str | None = None) -> dict:
        """Read completed trials and their recorded protocols, optionally restricted by evaluation mode."""
        completed = {}
        protocols = set()
        for run in self._search_runs(f"tags.`mgs_kan.series_key` = '{series_key}'"):
            tags = run.data.tags
            key = tags.get("mgs_kan.trial")
            if not key or (evaluation is not None and tags.get("mgs_kan.evaluation", "test") != evaluation):
                continue
            if protocol := tags.get("mgs_kan.protocol"):
                protocols.add(protocol)
            if key not in completed and self._is_complete(run):
                completed[key] = self._scores(run)
        return {"completed": completed, "protocols": sorted(protocols)}

    def matching_validation_trials(self, series: "ExperimentSeries", data: DatasetSplits) -> MatchingTrials:
        """Reuse the newest complete matching validation run across series in this experiment."""
        if series.evaluation != "validation" or series.folds != (data.fold,) or not data.source_sha256:
            raise ValueError("Cross-series reuse requires one validation fold and fingerprinted data")
        expected = {
            trial.key: ValidationRecipe.for_trial(series, trial, data)
            for trial in series.trials()
        }
        query = ("attributes.status = 'FINISHED' AND tags.`mgs_kan.evaluation` = 'validation' "
                 f"AND params.dataset_sha256 = '{data.source_sha256}' AND params.fold = '{data.fold}' "
                 f"AND params.n_features = '{data.n_features}'")
        completed = {}
        for run in self._search_runs(query):
            params = run.data.params
            key = "/".join(params.get(name, "") for name in ("model", "fold", "model.init_seed", "shuffle_seed"))
            recipe = expected.get(key)
            if recipe is None or key in completed or not recipe.matches(params):
                continue
            metrics = run.data.metrics
            epoch = metrics.get("best_epoch", math.nan)
            if not (1 <= epoch <= series.training.epochs and int(epoch) == epoch):
                continue
            if not all(0 <= metrics.get(k, math.nan) < math.inf for k in ("best_val_rmse", "duration_seconds")):
                continue
            if self._is_complete(run):
                completed[key] = self._scores(run)
                if len(completed) == len(expected):
                    break
        return MatchingTrials(data.source_sha256, completed)

    def validation_r2(self, run_id: str, epoch: int) -> float | None:
        """Read the R² logged at the selected checkpoint's one-based epoch."""
        history = self.client.get_metric_history(run_id, "val_r2")
        return next((metric.value for metric in history if metric.step == epoch), None)

    def grid_completed_counts(self, planned: dict[str, set[str]]) -> dict[str, int]:
        """Count reusable trials across the whole grid, including later series."""
        found = {key: set() for key in planned}
        for run in self._search_runs("attributes.status = 'FINISHED' AND tags.`mgs_kan.evaluation` = 'validation'"):
            tags = run.data.tags
            series = tags.get("mgs_kan.series_key")
            trial = tags.get("mgs_kan.trial")
            if series not in planned or trial not in planned[series] or trial in found[series]:
                continue
            if self._is_complete(run):
                found[series].add(trial)
        return {key: len(trials) for key, trials in found.items()}

    def _search_runs(self, filter_string: str):
        experiment_id = self._experiment_id()
        token = None
        while True:
            runs = self.client.search_runs(
                [experiment_id],
                filter_string=filter_string,
                page_token=token,
                order_by=["attributes.start_time DESC"],
            )
            yield from runs
            token = runs.token
            if not token:
                return

    def _is_complete(self, run) -> bool:
        if run.info.status != "FINISHED" or run.data.tags.get("mgs_kan.evaluated") != "true":
            return False
        score = "best_val_rmse" if run.data.tags.get("mgs_kan.evaluation") == "validation" else "test_rmse"
        if not {score, "best_epoch"}.issubset(run.data.metrics):
            return False
        filename = run.data.tags.get(_CHECKPOINT_TAG, DEFAULT_CHECKPOINT_FILE)
        files = self.client.list_artifacts(run.info.run_id)
        return any(file.path == filename and not file.is_dir for file in files)

    def _experiment_id(self) -> str:
        name = self.settings.experiment_name
        experiment = self.client.get_experiment_by_name(name)
        if experiment is None:
            try:
                return self.client.create_experiment(name)
            except MlflowException:
                experiment = self.client.get_experiment_by_name(name)
                if experiment is None:
                    raise
        return experiment.experiment_id

    @contextmanager
    def start_run(
        self,
        model: nn.Module,
        settings: TrainingSettings,
        data: DatasetSplits,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Iterator[MlflowRun]:
        params = {
            "model": getattr(model, "model_name", type(model).__name__),
            "trainable_parameters": parameter_count(model),
            "dataset": data.source_file or type(data).__name__,
            "dataset_sha256": data.source_sha256,
            "target_column": data.target_column,
            "input_protocol": data.input_protocol,
            "fold": data.fold,
            "n_features": data.n_features,
            "feature_columns": json.dumps(data.selected_features or data.columns) if data.columns else None,
            "seed": settings.seed,
            "shuffle_seed": settings.seed,
            "epochs": settings.epochs,
            "batch_size": settings.batch_size,
            "device": str(device),
            "optimizer": type(optimizer).__name__,
        }
        model_settings = getattr(model, "model_settings", None)
        if isinstance(model_settings, ModelSettings):
            for name, value in model_settings.model_dump().items():
                params[f"model.{name}"] = value
        params.update(optimizer_params(optimizer))

        filename = self.settings.checkpoint_file
        tags = {**self.run_tags, **SourceVersion().tags()}
        tags[_CHECKPOINT_TAG] = filename
        run = self.client.create_run(self._experiment_id(), tags=tags)
        run_id = run.info.run_id
        self.last_run_id = run_id
        try:
            self.client.log_batch(
                run_id, params=[Param(k, str(v)) for k, v in params.items() if v is not None]
            )
            yield MlflowRun(self.client, run_id, filename)
        except KeyboardInterrupt:
            self.client.set_terminated(run_id, status="KILLED")
            raise
        except Exception as error:
            try:
                self.client.set_terminated(run_id, status="FAILED")
                if self.run_tags:
                    self.client.set_tag(run_id, "mgs_kan.failure", str(error)[:1000])
            except Exception as tracking_error:
                error.add_note(f"Could not record the MLflow failure: {tracking_error}")
            raise
        else:
            self.client.set_terminated(run_id, status="FINISHED")
