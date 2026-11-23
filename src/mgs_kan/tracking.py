"""Record training runs on an existing MLflow tracking server."""

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time

import torch
from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from torch import nn

from .config import DEFAULT_CHECKPOINT_FILE, ModelSettings, TrackingSettings, TrainingSettings
from .data import DatasetSplits
from .models import parameter_count

_CHECKPOINT_TAG = "mgs_kan.checkpoint_file"


def _code_tags(package: Path) -> dict[str, str]:
    digest = sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    tags = {"mgs_kan.source_sha256": digest.hexdigest()}

    # An installed wheel may be inside an unrelated project's checkout.
    if package.parent.name == "src" and (package.parent.parent / ".git").exists():
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=package, stderr=subprocess.DEVNULL, text=True
            )
            status = subprocess.check_output(
                ["git", "--no-optional-locks", "status", "--porcelain", "--", "."],
                cwd=package, stderr=subprocess.DEVNULL, text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return tags
        tags["mlflow.source.git.commit"] = commit.strip()
        tags["mgs_kan.source_dirty"] = "true" if status else "false"
    return tags


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
    def __init__(self, settings: TrackingSettings | None = None):
        self.settings = settings or TrackingSettings()
        self.client = MlflowClient(tracking_uri=self.settings.tracking_uri)

    def download_checkpoint(self, run_id: str) -> bytes:
        tags = self.client.get_run(run_id).data.tags
        filename = tags.get(_CHECKPOINT_TAG, DEFAULT_CHECKPOINT_FILE)
        TrackingSettings.validate_checkpoint_file(filename)
        with TemporaryDirectory() as directory:
            path = self.client.download_artifacts(run_id, filename, directory)
            return Path(path).read_bytes()

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
        tags = _code_tags(Path(__file__).resolve().parent)
        tags[_CHECKPOINT_TAG] = filename
        run = self.client.create_run(self._experiment_id(), tags=tags)
        run_id = run.info.run_id
        try:
            self.client.log_batch(
                run_id, params=[Param(k, str(v)) for k, v in params.items() if v is not None]
            )
            yield MlflowRun(self.client, run_id, filename)
        except KeyboardInterrupt:
            self.client.set_terminated(run_id, status="KILLED")
            raise
        except Exception:
            self.client.set_terminated(run_id, status="FAILED")
            raise
        else:
            self.client.set_terminated(run_id, status="FINISHED")
