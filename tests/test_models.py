import os

import pytest
import torch
from pydantic import ValidationError

from mgs_kan.config import ModelSettings
from mgs_kan.models import make_model, parameter_count


@pytest.fixture(autouse=True)
def clear_model_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.upper().startswith("MGS_KAN_"):
            monkeypatch.delenv(key)


@pytest.mark.parametrize(("name", "count"), [("mlp", 937), ("mgs_kan", 949)])
def test_scalar_predictions(name, count):
    model = make_model(name)
    with torch.no_grad():
        prediction = model(torch.zeros(3, 12))
    assert parameter_count(model) == count
    assert prediction.shape == (3,)
    assert torch.isfinite(prediction).all()


@pytest.mark.parametrize("name", ["mlp", "mgs_kan"])
def test_feature_overrides(name, monkeypatch):
    monkeypatch.setenv("MGS_KAN_FEATURES", "5")
    with torch.no_grad():
        assert make_model(name)(torch.zeros(2, 5)).shape == (2,)
        settings = ModelSettings(features=3)
        assert make_model(name, settings)(torch.zeros(2, 3)).shape == (2,)


def test_invalid_kernel_count(monkeypatch):
    monkeypatch.setenv("MGS_KAN_PAIR_KERNELS", "0")
    with pytest.raises(ValidationError, match="pair_kernels"):
        make_model("mgs_kan")
