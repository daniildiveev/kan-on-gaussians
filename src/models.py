"""Scalar regressors. Model construction is independent of the target generator."""

import torch
from torch import nn

from . import config as c


class MGS_KAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pairs", torch.triu_indices(c.FEATURES, c.FEATURES, offset=1).T)
        singles = (c.FEATURES, c.SINGLE_KERNELS)
        pairs = (len(self.pairs), c.PAIR_KERNELS)
        self.affine = nn.Linear(c.FEATURES, 1)
        self.single_center = nn.Parameter(
            torch.linspace(*c.SINGLE_CENTER_RANGE, c.SINGLE_KERNELS).repeat(c.FEATURES, 1)
        )
        self.single_amplitude = nn.Parameter(c.SINGLE_AMPLITUDE_SD * torch.randn(singles))
        self.single_log_scale = nn.Parameter(torch.zeros(singles))
        self.pair_center = nn.Parameter(c.PAIR_CENTER_SD * torch.randn(*pairs, 2))
        self.pair_amplitude = nn.Parameter(c.PAIR_AMPLITUDE_SD * torch.randn(pairs))
        self.pair_precision_raw = nn.Parameter(c.PAIR_FACTOR_SD * torch.randn(*pairs, 3))

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


def make_model(name):
    if name == "mgs_kan":
        return MGS_KAN()
    if name != "mlp":
        raise ValueError(f"Unknown model {name!r}")
    return nn.Sequential(
        nn.Linear(c.FEATURES, c.MLP_WIDTH), nn.ReLU(),
        nn.Linear(c.MLP_WIDTH, c.MLP_WIDTH), nn.ReLU(),
        nn.Linear(c.MLP_WIDTH, 1), nn.Flatten(start_dim=0),
    )


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
