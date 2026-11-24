"""Portable snapshots of the model and optimizer at the best validation epoch."""

from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from src.paths import project_path


def _class_name(obj) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _parameter_groups(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[list[str]]:
    names = {param: name for name, param in model.named_parameters()}
    try:
        return [[names[p] for p in group["params"]] for group in optimizer.param_groups]
    except KeyError as error:
        raise ValueError("Optimizer must reference parameters of the trainer's model") from error


@dataclass(frozen=True)
class Checkpoint:
    """Serialized best-epoch weights and optimizer state."""

    content: bytes

    @classmethod
    def capture(cls, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> "Checkpoint":
        snapshot = {
            "format_version": 1,
            "best_epoch": epoch,
            "model_class": _class_name(model),
            "model_state": model.state_dict(),
            "optimizer_class": _class_name(optimizer),
            "parameter_groups": _parameter_groups(model, optimizer),
            "optimizer_state": optimizer.state_dict(),
        }
        with BytesIO() as buffer:
            torch.save(snapshot, buffer)
            return cls(buffer.getvalue())

    def restore(
        self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        snapshot = torch.load(BytesIO(self.content), map_location="cpu", weights_only=True)
        if not isinstance(snapshot, dict) or snapshot.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        epoch = snapshot["best_epoch"]
        saved_class = snapshot["model_class"]
        if saved_class in ("models.mlp.MLP", "models.kan.MGS_KAN"):
            saved_class = f"src.{saved_class}"
        if saved_class != _class_name(model):
            raise ValueError("Checkpoint requires the same model class")
        if optimizer is not None:
            if snapshot["optimizer_class"] != _class_name(optimizer):
                raise ValueError("Checkpoint requires the same optimizer class")
            if snapshot["parameter_groups"] != _parameter_groups(model, optimizer):
                raise ValueError("Optimizer parameter groups and their order must match the checkpoint")

        # A state_dict loader may copy some tensors before reporting an incompatibility.
        previous_weights = deepcopy(model.state_dict())
        previous_optimizer = deepcopy(optimizer.state_dict()) if optimizer is not None else None
        try:
            model.load_state_dict(snapshot["model_state"])
            if optimizer is not None:
                optimizer.load_state_dict(snapshot["optimizer_state"])
        except Exception:
            model.load_state_dict(previous_weights)
            if optimizer is not None:
                optimizer.load_state_dict(previous_optimizer)
            raise
        model.eval()
        return epoch

    def save(self, path: str | Path) -> None:
        path = project_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as directory:
            staged = Path(directory) / "checkpoint.pt"
            staged.write_bytes(self.content)
            staged.replace(path)
