from typing import Literal

from torch import nn

from src.config import ModelSettings
from src.models.hypergraph import HypergraphMGS_KAN
from src.models.kan import MGS_KAN
from src.models.mlp import DeepMLP, MLP
from src.models.resnet import TabularResNet


ModelName = Literal["mlp", "mgs_kan", "deep_mlp", "resnet", "hypergraph_mgs_kan"]


def make_model(name: str, settings: ModelSettings | None = None) -> nn.Module:
    if name == "mlp":
        return MLP(settings)
    if name == "mgs_kan":
        return MGS_KAN(settings)
    if name == "hypergraph_mgs_kan":
        return HypergraphMGS_KAN(settings)
    if name == "deep_mlp":
        return DeepMLP(settings)
    if name == "resnet":
        return TabularResNet(settings)
    raise ValueError(f"Unknown model {name!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
