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
        width = self.model_settings.mlp_width
        with initialize_weights(self, self.model_settings.init_seed):
            self.extend([
                nn.Linear(self.model_settings.features, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, 1),
                nn.Flatten(start_dim=0),
            ])

    def __getitem__(self, index):
        if isinstance(index, slice):
            # A slice is a view of the selected layers, with the original parameter names.
            return nn.Sequential(OrderedDict(list(self._modules.items())[index]))
        return super().__getitem__(index)
