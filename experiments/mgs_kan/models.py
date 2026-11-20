"""Scalar regressors at a matched trainable parameter budget."""

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
            torch.linspace(-1.5, 1.5, c.SINGLE_KERNELS).repeat(c.FEATURES, 1)
        )
        self.single_amplitude = nn.Parameter(0.09 * torch.randn(singles))
        self.single_log_scale = nn.Parameter(torch.zeros(singles))
        self.pair_center = nn.Parameter(0.55 * torch.randn(*pairs, 2))
        self.pair_amplitude = nn.Parameter(0.035 * torch.randn(pairs))
        self.pair_precision_raw = nn.Parameter(0.08 * torch.randn(*pairs, 3))

    def factor_entries(self):
        log_a, b, log_c = self.pair_precision_raw.unbind(dim=-1)
        return log_a.exp(), b, log_c.exp()

    def precision(self):
        a, b, d = self.factor_entries()
        return torch.stack((a.square(), a*b, a*b, b.square() + d.square()), dim=-1).unflatten(-1, (2, 2))

    def pair_coordinates(self, x):
        return (x[:, self.pairs].unsqueeze(2) - self.pair_center).unbind(dim=-1)

    def pair_kernels(self, u, v):
        a, b, d = self.factor_entries()
        # For P = L L^T, evaluate ||L^T (x - center)||^2.
        return torch.exp(-0.5 * ((a*u + b*v).square() + (d*v).square()))

    def forward(self, x):
        single_delta = (x.unsqueeze(-1) - self.single_center) * self.single_log_scale.exp()
        singles = torch.exp(-0.5 * single_delta.square()) * self.single_amplitude
        pairs = self.pair_kernels(*self.pair_coordinates(x)) * self.pair_amplitude
        return self.affine(x).squeeze(-1) + singles.sum(dim=(1, 2)) + pairs.sum(dim=(1, 2))

    def group_penalty(self):
        return torch.linalg.vector_norm(self.pair_amplitude, dim=1).sum()

    def pure_interactions(self, x, mean, sd):
        """Per-pair second-order ANOVA component under independent Gaussian inputs."""
        u, v = self.pair_coordinates(x)
        mu, nu = (mean[self.pairs].unsqueeze(1) - self.pair_center).unbind(dim=-1)
        su, sv = sd[self.pairs].square().unsqueeze(1).unbind(dim=-1)
        a, b, d = self.factor_entries()
        p, q, r = a.square(), a*b, b.square() + d.square()
        determinant = (a*d).square()
        du, dv = 1 + su*p, 1 + sv*r
        joint = 1 + su*p + sv*r + su*sv*determinant
        # Integrate each kernel by completing the square in the marginalized coordinate.
        e_given_u = torch.exp(-0.5 * ((p + sv*determinant)*u.square() + 2*q*u*nu + r*nu.square()) / dv) / dv.sqrt()
        e_given_v = torch.exp(-0.5 * (p*mu.square() + 2*q*mu*v + (r + su*determinant)*v.square()) / du) / du.sqrt()
        e_joint = torch.exp(-0.5 * ((p + sv*determinant)*mu.square() + 2*q*mu*nu
                                  + (r + su*determinant)*nu.square()) / joint) / joint.sqrt()
        pure = self.pair_kernels(u, v) - e_given_u - e_given_v + e_joint
        return (pure * self.pair_amplitude).sum(dim=-1)


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
