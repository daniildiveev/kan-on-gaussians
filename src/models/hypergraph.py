"""Scalar MGS-KAN with one full-covariance Gaussian per unordered feature triple."""

from itertools import combinations

import torch
from torch import Tensor, nn

from src.config import ModelSettings
from src.models.kan import MGS_KAN


class HypergraphMGS_KAN(MGS_KAN):
    model_name = "hypergraph_mgs_kan"

    def _init_parameters(self, settings: ModelSettings) -> None:
        if settings.features < 3:
            raise ValueError("Third-order MGS-KAN requires at least three features")
        super()._init_parameters(settings)
        triples = list(combinations(range(settings.features), 3))
        self.register_buffer("triples", torch.tensor(triples, dtype=torch.long))
        count = len(triples)
        self.triple_center = nn.Parameter(settings.pair_center_sd * torch.randn(count, 3))
        self.triple_amplitude = nn.Parameter(settings.pair_amplitude_sd * torch.randn(count))
        self.triple_factor_raw = nn.Parameter(settings.pair_factor_sd * torch.randn(count, 6))

    def triple_values(self, coordinates: Tensor) -> Tensor:
        """Evaluate each Gaussian at its three coordinates, returning [..., n_triples]."""
        x, y, z = (coordinates - self.triple_center).unbind(-1)
        log_a, b, log_c, d, e, log_f = self.triple_factor_raw.unbind(-1)
        # Lower triangular precision factor L; exponent is -||Lᵀ (x-center)||² / 2.
        squared = ((x * log_a.exp() + y * b + z * d).square()
                   + (y * log_c.exp() + z * e).square() + (z * log_f.exp()).square())
        return self.triple_amplitude * torch.exp(-0.5 * squared)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) + self.triple_values(x[..., self.triples]).sum(-1)
