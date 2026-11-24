"""Record training runs on an existing MLflow tracking server."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time

import torch
from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from torch import nn

from config import DEFAULT_CHECKPOINT_FILE, ModelSettings, TrackingSettings, TrainingSettings
from data import DatasetSplits
from models import parameter_count

from .provenance import SourceVersion

_CHECKPOINT_TAG = "mgs_kan.checkpoint_file"


def _optimizer_params(optimizer: torch.optim.Optimizer) -> dict[str, str]:
    params = {}
    for index, group in enumerate(optimizer.param_groups):
        for name, value in group.items():
            if name in ("params", "param_names"):
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().tolist()
            params[f"optimizer.{name}.{index}"] = json.dumps(value, default=str)
    return params


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
            "test_rmse": metrics["test_rmse"],
            "test_r2": metrics.get("test_r2"),
        }

    def completed_trials(self, series_key: str) -> dict:
        """Read reusable evaluations and protocol IDs for this series."""
        experiment_id = self._experiment_id()
        completed = {}
        protocols = set()
        token = None
        while True:
            runs = self.client.search_runs(
                [experiment_id],
                filter_string=f"tags.`mgs_kan.series_key` = '{series_key}'",
                page_token=token,
                order_by=["attributes.start_time DESC"],
            )
            for run in runs:
                tags = run.data.tags
                key = tags.get("mgs_kan.trial")
                protocol = tags.get("mgs_kan.protocol")
                # FileStore may be interrupted halfway through creating a tagged run.
                if not key or not protocol:
                    continue
                protocols.add(protocol)
                if key not in completed and self._is_complete(run):
                    completed[key] = self._scores(run)
            token = runs.token
            if not token:
                return {"completed": completed, "protocols": sorted(protocols)}

    def _is_complete(self, run) -> bool:
        if run.info.status != "FINISHED" or run.data.tags.get("mgs_kan.evaluated") != "true":
            return False
        if not {"test_rmse", "best_epoch"}.issubset(run.data.metrics):
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
            "fold": data.fold,
            "n_features": data.n_features,
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
        params.update(_optimizer_params(optimizer))

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
