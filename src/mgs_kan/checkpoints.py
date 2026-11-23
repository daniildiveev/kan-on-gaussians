"""Portable snapshots of the model and optimizer at the best validation epoch."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn


def _class_name(obj) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _parameter_groups(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[list[str]]:
    names = {param: name for name, param in model.named_parameters()}
    try:
        return [[names[p] for p in group["params"]] for group in optimizer.param_groups]
    except KeyError as error:
        raise ValueError("Optimizer must reference parameters of the trainer's model") from error


def encode_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> bytes:
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
        return buffer.getvalue()


def restore_checkpoint(
    content: bytes, model: nn.Module, optimizer: torch.optim.Optimizer | None,
) -> int:
    snapshot = torch.load(BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(snapshot, dict) or snapshot.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    epoch = snapshot["best_epoch"]
    if snapshot["model_class"] != _class_name(model):
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


def write_checkpoint(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / "checkpoint.pt"
        staged.write_bytes(content)
        staged.replace(path)
