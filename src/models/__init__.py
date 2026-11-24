"""Scalar regression models."""

from .factory import make_model, parameter_count
from .kan import MGS_KAN
from .mlp import MLP

__all__ = ["MLP", "MGS_KAN", "make_model", "parameter_count"]
