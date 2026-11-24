from torch import nn

from config import ModelSettings
from .kan import MGS_KAN
from .mlp import MLP


def make_model(name: str, settings: ModelSettings | None = None) -> nn.Module:
    if name == "mlp":
        return MLP(settings)
    if name == "mgs_kan":
        return MGS_KAN(settings)
    raise ValueError(f"Unknown model {name!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
