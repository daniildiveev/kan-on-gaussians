"""Scalar regressors. Model construction is independent of the target generator."""

from contextlib import contextmanager

import torch
from torch import nn

from .config import ModelSettings


@contextmanager
def _initialize(model: nn.Module, seed: int | None):
    if seed is None:
        yield
        return

    # Seed only CPU initialization, then place weights on the caller's default device.
    device = torch.empty(0).device
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.default_generator.manual_seed(seed)
        yield
    if device.type != "cpu":
        model.to(device)


class MGS_KAN(nn.Module):
    model_name = "mgs_kan"

    def __init__(self, settings: ModelSettings | None = None):
        super().__init__()
        if settings is None:
            settings = ModelSettings()
        self.model_settings = settings
        with _initialize(self, settings.init_seed):
            self._init_parameters(settings)

    def _init_parameters(self, settings: ModelSettings) -> None:
        self.register_buffer("pairs", torch.triu_indices(settings.features, settings.features, offset=1).T)
        singles = (settings.features, settings.single_kernels)
        pairs = (len(self.pairs), settings.pair_kernels)
        self.affine = nn.Linear(settings.features, 1)
        self.single_center = nn.Parameter(
            torch.linspace(*settings.single_center_range, settings.single_kernels).repeat(settings.features, 1)
        )
        self.single_amplitude = nn.Parameter(settings.single_amplitude_sd * torch.randn(singles))
        self.single_log_scale = nn.Parameter(torch.zeros(singles))
        self.pair_center = nn.Parameter(settings.pair_center_sd * torch.randn(*pairs, 2))
        self.pair_amplitude = nn.Parameter(settings.pair_amplitude_sd * torch.randn(pairs))
        self.pair_precision_raw = nn.Parameter(settings.pair_factor_sd * torch.randn(*pairs, 3))

    def factor_entries(self):
        log_a, b, log_d = self.pair_precision_raw.unbind(dim=-1)
        return log_a.exp(), b, log_d.exp()

    def precision(self):
        a, b, d = self.factor_entries()
        factor = torch.stack((a, torch.zeros_like(a), b, d), dim=-1).unflatten(-1, (2, 2))
        return factor @ factor.mT

    def pair_values(self, u, v, pair=slice(None)):
        """Sum kernels per pair at broadcastable coordinates with the pair axis last."""
        a, b, d = (entry[pair] for entry in self.factor_entries())
        center_u, center_v = self.pair_center[pair].unbind(dim=-1)
        du, dv = u.unsqueeze(-1) - center_u, v.unsqueeze(-1) - center_v
        # For P = L Lᵀ, dᵀ P d = ||Lᵀ d||².
        kernels = torch.exp(-0.5 * ((a * du + b * dv).square() + (d * dv).square()))
        return (kernels * self.pair_amplitude[pair]).sum(dim=-1)

    def contributions(self, x):
        """Return the affine term, one term per feature, and one term per pair."""
        delta = (x.unsqueeze(-1) - self.single_center) * self.single_log_scale.exp()
        singles = torch.exp(-0.5 * delta.square()) * self.single_amplitude
        pairs = self.pair_values(x[:, self.pairs[:, 0]], x[:, self.pairs[:, 1]])
        return self.affine(x).squeeze(-1), singles.sum(dim=-1), pairs

    def forward(self, x):
        affine, singles, pairs = self.contributions(x)
        return affine + singles.sum(dim=-1) + pairs.sum(dim=-1)

    def group_penalty(self):
        return torch.linalg.vector_norm(self.pair_amplitude, dim=1).sum()


def make_model(name: str, settings: ModelSettings | None = None):
    if name == "mgs_kan":
        return MGS_KAN(settings)
    if name != "mlp":
        raise ValueError(f"Unknown model {name!r}")
    if settings is None:
        settings = ModelSettings()
    model = nn.Sequential()
    with _initialize(model, settings.init_seed):
        model.extend([
            nn.Linear(settings.features, settings.mlp_width), nn.ReLU(),
            nn.Linear(settings.mlp_width, settings.mlp_width), nn.ReLU(),
            nn.Linear(settings.mlp_width, 1), nn.Flatten(start_dim=0),
        ])
    model.model_name = "mlp"
    model.model_settings = settings
    return model


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
