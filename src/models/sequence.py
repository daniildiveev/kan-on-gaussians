"""Scalar regression from a fixed window of sensor observations."""

from torch import Tensor, nn

from src.config import GAS_TURBINE_WINDOW_SIZE, ModelSettings
from src.models.initialization import initialize_weights
from src.models.mlp import MLP


class WindowRegressor(nn.Module):
    def __init__(self, settings: ModelSettings | None = None):
        super().__init__()
        self.model_settings = settings or ModelSettings()

    def check_window(self, x: Tensor) -> None:
        shape = (GAS_TURBINE_WINDOW_SIZE, self.model_settings.features)
        if x.ndim != 3 or x.shape[1:] != shape:
            raise ValueError(f"Expected a batch of windows with shape {shape}")


class WindowGRU(WindowRegressor):
    model_name = "gru_window6"

    def __init__(self, settings: ModelSettings | None = None):
        super().__init__(settings)
        width = self.model_settings.deep_width
        with initialize_weights(self, self.model_settings.init_seed):
            self.gru = nn.GRU(self.model_settings.features, width, batch_first=True)
            self.readout = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        self.check_window(x)
        _, hidden = self.gru(x)
        return self.readout(hidden[-1]).squeeze(-1)


class WindowMLP(WindowRegressor):
    model_name = "mlp_window6"

    def __init__(self, settings: ModelSettings | None = None):
        super().__init__(settings)
        inputs = self.model_settings.features * GAS_TURBINE_WINDOW_SIZE
        self.mlp = MLP(self.model_settings.model_copy(update={"features": inputs}))

    def forward(self, x: Tensor) -> Tensor:
        self.check_window(x)
        return self.mlp(x.flatten(start_dim=1))
