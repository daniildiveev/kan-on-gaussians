from collections import OrderedDict

from torch import nn

from src.config import ModelSettings
from src.models.initialization import initialize_weights


class MLP(nn.Sequential):
    """Two ReLU hidden layers and one scalar output per example."""

    model_name = "mlp"

    def __init__(self, settings: ModelSettings | None = None):
        super().__init__()
        self.model_settings = settings or ModelSettings()
        with initialize_weights(self, self.model_settings.init_seed):
            inputs = self.model_settings.features
            for width in self.hidden_widths():
                self.extend([nn.Linear(inputs, width), nn.ReLU()])
                inputs = width
            self.extend([nn.Linear(inputs, 1), nn.Flatten(start_dim=0)])

    def hidden_widths(self) -> tuple[int, ...]:
        return (self.model_settings.mlp_width,) * 2

    def __getitem__(self, index):
        if isinstance(index, slice):
            # A slice is a view of the selected layers, with the original parameter names.
            return nn.Sequential(OrderedDict(list(self._modules.items())[index]))
        return super().__getitem__(index)


class DeepMLP(MLP):
    model_name = "deep_mlp"

    def hidden_widths(self) -> tuple[int, ...]:
        return (self.model_settings.deep_width,) * self.model_settings.deep_depth
