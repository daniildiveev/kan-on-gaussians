import math

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.utils.data import TensorDataset

from src.config import ModelSettings, TrainingSettings
from src.data import DatasetSplits
from src.models import make_model
from src.training import Trainer, evaluate
from src.training.trainer import _resolve_device


@pytest.mark.parametrize(("cuda", "mps", "expected"), [
    (True, True, "cuda"),
    (False, True, "mps"),
    (False, False, "cpu"),
])
def test_auto_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert _resolve_device("auto") == torch.device(expected)
    assert _resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("name", ["cuda", "cuda:0", "mps"])
def test_unavailable_explicit_device_raises(monkeypatch, name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match=name.split(":")[0]):
        _resolve_device(name)


def test_cuda_index_uses_visible_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert _resolve_device("cuda:1") == torch.device("cuda:1")
    with pytest.raises(ValueError, match="cuda:2"):
        _resolve_device("cuda:2")


def test_device_configuration(monkeypatch):
    monkeypatch.delenv("MGS_KAN_TRAIN_DEVICE", raising=False)
    assert TrainingSettings().device == "cpu"
    monkeypatch.setenv("MGS_KAN_TRAIN_DEVICE", "cuda:1")
    assert TrainingSettings().device == "cuda:1"
    with pytest.raises(ValidationError):
        TrainingSettings(device="cuda:-1")


def test_evaluate_parameterless_model():
    x = torch.arange(4.0)
    split = TensorDataset(x[:, None], x)
    data = DatasetSplits(split, split, split, 1, 0.0, 1.0)
    assert evaluate(nn.Flatten(start_dim=0), data) == 0.0


@pytest.mark.parametrize("name", ["mlp", "mgs_kan"])
@pytest.mark.parametrize("use_rmsprop", [False, True], ids=["default-adam", "rmsprop"])
@pytest.mark.parametrize("backend", [
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")),
])
def test_fit_and_evaluate_on_gpu(name, backend, use_rmsprop):
    x = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
    y = x.sum(dim=1)
    split = TensorDataset(x, y)
    data = DatasetSplits(split, split, split, 2, 20.0, 3.0)
    model = make_model(name, ModelSettings(features=2))
    original = next(model.parameters()).detach().clone()

    settings = TrainingSettings(device=backend, epochs=2, batch_size=3)
    trainer = Trainer(model, settings)
    optimizer = torch.optim.RMSprop(trainer.model.parameters(), lr=0.002) if use_rmsprop else None
    result = trainer.fit(data, optimizer=optimizer)
    assert math.isfinite(min(result.validation_rmse))
    assert all(p.device.type == backend for p in model.parameters())
    assert all(b.device.type == backend for b in model.buffers())
    assert not torch.equal(next(model.parameters()).cpu(), original)
    assert all(t.device.type == "cpu" for t in data.train.tensors)

    predictions = trainer.predict(x)
    assert predictions.device.type == backend
    error = data.inverse_target(predictions.cpu().double()) - data.inverse_target(y.double())
    expected = error.square().mean().sqrt().item()
    assert evaluate(model, data, batch_size=3) == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert trainer.evaluate(data) == pytest.approx(expected, rel=1e-5, abs=1e-6)
