"""Scalar regression models."""

from src.models.factory import make_model, parameter_count
from src.models.kan import MGS_KAN
from src.models.mlp import MLP

__all__ = ["MLP", "MGS_KAN", "make_model", "parameter_count"]
