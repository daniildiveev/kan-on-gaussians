from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def initialize_weights(model: nn.Module, seed: int | None):
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
