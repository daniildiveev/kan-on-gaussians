"""Training and held-out evaluation for scalar regression."""

from __future__ import annotations

import math
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoints import encode_checkpoint, restore_checkpoint, write_checkpoint
from .config import TrainingSettings
from .data import DatasetSplits

if TYPE_CHECKING:
    from .tracking import MlflowRun, MlflowTracker


@dataclass(frozen=True)
class FitResult:
    """One entry per epoch. best_epoch is one-based."""

    best_epoch: int
    train_mse: tuple[float, ...]
    validation_rmse: tuple[float, ...]
    run_id: str | None = None


def _r2_metric(name: str, rmse: float, variance: float) -> dict[str, float]:
    if math.isnan(variance):  # A single observation has no R².
        return {}
    if variance == 0:
        return {name: float(rmse == 0)}
    return {name: 1.0 - rmse**2 / variance}


def _target_variance(targets: torch.Tensor, data: DatasetSplits) -> float:
    if len(targets) < 2:
        return math.nan
    values = data.inverse_target(targets.cpu().double())
    return (values - values[0]).var(correction=0).item()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested mps device is unavailable")

    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested {name} device is unavailable")
        _, _, index = name.partition(":")
        if index and int(index) >= torch.cuda.device_count():
            raise ValueError(f"Requested {name} is outside the visible CUDA device range")

    return torch.device(name)


class Trainer:
    """Train and evaluate one scalar regressor in place."""

    def __init__(
        self,
        model: nn.Module,
        settings: TrainingSettings | None = None,
        *,
        tracker: MlflowTracker | None = None,
    ):
        self.model = model
        self.settings = settings
        self.tracker = tracker
        self._last_run: MlflowRun | None = None
        self._fitted_data: DatasetSplits | None = None
        self._best_epoch = 0
        self._checkpoint: bytes | None = None
        if settings is None:
            tensor = next(chain(model.parameters(), model.buffers()), None)
            self.device = torch.device("cpu") if tensor is None else tensor.device
        else:
            self._place_model(settings)

    def _place_model(self, settings: TrainingSettings) -> None:
        self.device = _resolve_device(settings.device)
        self.model.to(self.device)

    def _forward(self, features: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        predictions = self.model(features.to(self.device))
        if predictions.shape == (len(features), 1):
            predictions = predictions.squeeze(-1)
        if predictions.shape != (len(features),):
            raise ValueError("Expected one scalar prediction per example")
        if targets is not None and targets.shape != predictions.shape:
            raise ValueError("Expected one scalar target per example")
        return predictions

    @torch.inference_mode()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Return one standardized prediction per row on the model's device."""
        self.model.eval()
        return self._forward(features)

    def _train_epoch(self, batches: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_squared_error = 0.0
        for features, targets in batches:
            optimizer.zero_grad(set_to_none=True)
            predictions = self._forward(features, targets)
            loss = F.mse_loss(predictions, targets.to(self.device))
            batch_mse = loss.item()
            if not math.isfinite(batch_mse):
                raise ValueError("Nonfinite training MSE")
            loss.backward()
            optimizer.step()

            total_squared_error += batch_mse * len(targets)

        return total_squared_error / len(batches.dataset)

    @torch.inference_mode()
    def _rmse(self, batches: DataLoader, data: DatasetSplits) -> float:
        self.model.eval()
        metric_device = torch.device("cpu") if self.device.type == "mps" else self.device
        total_squared_error = 0.0
        for features, targets in batches:
            predictions = self._forward(features, targets).to(metric_device).double()
            targets = targets.to(metric_device).double()
            predicted_values = data.inverse_target(predictions)
            actual_values = data.inverse_target(targets)
            total_squared_error += (predicted_values - actual_values).square().sum().item()

        rmse = math.sqrt(total_squared_error / len(batches.dataset))
        if not math.isfinite(rmse):
            raise ValueError("Nonfinite RMSE")
        return rmse

    def fit(
        self,
        data: DatasetSplits,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> FitResult:
        """Train with the supplied optimizer or a fresh Adam using settings.learning_rate.

        Restore model and optimizer state from the earliest minimum validation RMSE.
        """
        if self.settings is None:
            settings = TrainingSettings()
            self._place_model(settings)
            self.settings = settings
        settings = self.settings

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=settings.learning_rate)
        else:
            model_parameters = set(self.model.parameters())
            for group in optimizer.param_groups:
                if not model_parameters.issuperset(group["params"]):
                    raise ValueError(
                        "Optimizer must reference model parameters on the training device. "
                        "Build it from trainer.model.parameters() after placing the model."
                    )

        self._last_run = None
        self._fitted_data = None
        self._checkpoint = None
        context = (
            self.tracker.start_run(self.model, settings, data, optimizer, self.device)
            if self.tracker is not None else nullcontext(None)
        )
        with context as run:
            result = self._fit_epochs(data, optimizer, settings, run)
            checkpoint = encode_checkpoint(self.model, optimizer, result.best_epoch)
            if run is not None:
                run.upload_checkpoint(checkpoint)
        self._checkpoint = checkpoint
        if run is not None:
            self._last_run = run
            self._fitted_data = data
            self._best_epoch = result.best_epoch
        return result

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the best epoch from the last successful fit or load."""
        if self._checkpoint is None:
            raise RuntimeError("Fit the model or load a checkpoint before saving")
        write_checkpoint(Path(path), self._checkpoint)

    def load_checkpoint(
        self,
        path: str | Path | None = None,
        *,
        run_id: str | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        """Restore weights and optional optimizer in place. Return the best epoch."""
        if (path is None) == (run_id is None):
            raise ValueError("Specify exactly one of path or run_id")
        if run_id is not None:
            if self.tracker is None:
                raise ValueError("Loading by run_id requires an MLflow tracker")
            content = self.tracker.download_checkpoint(run_id)
        else:
            content = Path(path).read_bytes()

        epoch = restore_checkpoint(content, self.model, optimizer)
        self._checkpoint = content
        self._best_epoch = epoch
        self._last_run = None
        self._fitted_data = None
        return epoch

    def _fit_epochs(
        self,
        data: DatasetSplits,
        optimizer: torch.optim.Optimizer,
        settings: TrainingSettings,
        run: MlflowRun | None,
    ) -> FitResult:
        order = torch.Generator().manual_seed(settings.seed)
        train_batches = DataLoader(
            data.train,
            batch_size=settings.batch_size,
            shuffle=True,
            generator=order,
        )
        validation_batches = DataLoader(data.val, batch_size=settings.batch_size)
        train_history = []
        validation_history = []
        best_rmse = math.inf
        if run is not None:
            variance = _target_variance(data.val.tensors[1], data)

        for epoch in range(1, settings.epochs + 1):
            train_history.append(self._train_epoch(train_batches, optimizer))
            validation_rmse = self._rmse(validation_batches, data)
            validation_history.append(validation_rmse)
            if run is not None:
                run.log_metrics(
                    epoch,
                    train_mse=train_history[-1],
                    val_rmse=validation_rmse,
                    **_r2_metric("val_r2", validation_rmse, variance),
                )
            if validation_rmse < best_rmse:
                best_rmse = validation_rmse
                best_epoch = epoch
                best_weights = deepcopy(self.model.state_dict())
                best_optimizer_state = deepcopy(optimizer.state_dict())

        self.model.load_state_dict(best_weights)
        optimizer.load_state_dict(best_optimizer_state)
        if run is not None:
            run.log_metrics(best_epoch, best_epoch=best_epoch)
        return FitResult(
            best_epoch=best_epoch,
            train_mse=tuple(train_history),
            validation_rmse=tuple(validation_history),
            run_id=run.run_id if run is not None else None,
        )

    def evaluate(self, data: DatasetSplits, *, batch_size: int | None = None) -> float:
        """Report test RMSE in original target units and leave the model in eval mode."""
        if batch_size is None:
            batch_size = self.settings.batch_size if self.settings is not None else 64
        test_batches = DataLoader(data.test, batch_size=batch_size)
        rmse = self._rmse(test_batches, data)
        if self._last_run is not None and self._fitted_data is data:
            variance = _target_variance(data.test.tensors[1], data)
            self._last_run.log_metrics(
                self._best_epoch, test_rmse=rmse, **_r2_metric("test_r2", rmse, variance)
            )
        return rmse


def fit(
    model: nn.Module,
    data: DatasetSplits,
    settings: TrainingSettings,
    optimizer: torch.optim.Optimizer | None = None,
) -> FitResult:
    """Fit the supplied model in place with a new trainer."""
    return Trainer(model, settings).fit(data, optimizer=optimizer)


def evaluate(model: nn.Module, data: DatasetSplits, *, batch_size: int = 64) -> float:
    """Evaluate on the model's current device and report RMSE in original target units."""
    return Trainer(model).evaluate(data, batch_size=batch_size)
