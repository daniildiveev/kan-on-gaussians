"""Residual MLP for standardized numeric tables."""

from torch import Tensor, nn

from src.config import ModelSettings
from src.models.initialization import initialize_weights


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.branch(x)


class TabularResNet(nn.Module):
    model_name = "resnet"

    def __init__(self, settings: ModelSettings | None = None):
        super().__init__()
        self.model_settings = settings or ModelSettings()
        width = self.model_settings.deep_width
        with initialize_weights(self, self.model_settings.init_seed):
            self.network = nn.Sequential(
                nn.Linear(self.model_settings.features, width),
                *(ResidualBlock(width) for _ in range(self.model_settings.deep_depth)),
                nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1)
