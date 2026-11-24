from copy import deepcopy
from math import sqrt

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.utils.data import TensorDataset

from config import ModelSettings, TrainingSettings
from data import CsvSpec, DatasetSplits, load_dataset
from models import make_model
from training import Trainer, evaluate, fit


@pytest.mark.parametrize("name", ["mlp", "mgs_kan"])
def test_repeatable_training_on_local_csv(name, tmp_path):
    x = np.arange(30)
    table = np.column_stack((x, x % 4, 2 * x + 1))
    np.savetxt(tmp_path / "small.csv", table, delimiter=",", header="a,b,y", comments="")
    data = load_dataset(CsvSpec("small.csv", "y"), root=tmp_path)
    settings = TrainingSettings(seed=71, epochs=3, batch_size=5, learning_rate=0.01)
    model = make_model(name, ModelSettings(features=data.n_features))
    replica = deepcopy(model)

    result = fit(model, data, settings)
    assert result == fit(replica, data, settings)
    assert len(result.train_mse) == len(result.validation_rmse) == settings.epochs
    assert result.best_epoch == 1 + result.validation_rmse.index(min(result.validation_rmse))
    assert evaluate(model, data) == evaluate(replica, data)


def test_fit_restores_earlier_validation_winner():
    x = torch.zeros(4, 1)
    train = TensorDataset(x, torch.ones(4))
    val = TensorDataset(x, -torch.ones(4))
    # Accessing the held-out split during fit would produce a nonfinite metric.
    test = TensorDataset(x, torch.full((4,), float("nan")))
    data = DatasetSplits(train, val, test, 1, 25.0, 3.0)
    model = nn.Sequential(nn.Linear(1, 1), nn.Flatten(start_dim=0))
    nn.init.zeros_(model[0].bias)
    first_epoch_model = deepcopy(model)
    settings = TrainingSettings(seed=7, epochs=4, batch_size=4, learning_rate=0.1)

    fit(first_epoch_model, data, settings.model_copy(update={"epochs": 1}))
    result = fit(model, data, settings)
    assert result.best_epoch == 1
    assert result.validation_rmse[-1] > result.validation_rmse[0]
    assert result.train_mse[-1] < result.train_mse[0]
    torch.testing.assert_close(model(x), first_epoch_model(x), rtol=0, atol=0)
    assert result.validation_rmse[0] == pytest.approx(3.3)
    assert not model.training


def test_fit_restores_and_reuses_supplied_optimizer(tmp_path):
    x = torch.ones(4, 1)
    train = TensorDataset(x, torch.ones(4))
    val = TensorDataset(x, -torch.ones(4))
    data = DatasetSplits(train, val, val, 1, 0.0, 1.0)
    settings = TrainingSettings(device="cpu", epochs=4, batch_size=4, learning_rate=0.5)
    model = nn.Linear(1, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    trainer = Trainer(model, settings)
    reference = deepcopy(model)

    def rmsprop(layer):
        return torch.optim.RMSprop(
            [{"params": [layer.weight], "lr": 0.01}, {"params": [layer.bias], "lr": 0.02}],
            momentum=0.2,
        )

    optimizer = rmsprop(model)
    reference_optimizer = rmsprop(reference)

    def check_against_next_reference_step():
        reference_optimizer.zero_grad()
        (reference(x).squeeze(-1) - train.tensors[1]).square().mean().backward()
        reference_optimizer.step()
        torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
        actual = optimizer.state_dict()
        expected = reference_optimizer.state_dict()
        assert actual["param_groups"] == expected["param_groups"]
        torch.testing.assert_close(actual["state"], expected["state"], rtol=0, atol=0)

    result = trainer.fit(data, optimizer=optimizer)
    assert result.best_epoch == 1
    assert result.validation_rmse[0] < result.validation_rmse[-1]
    check_against_next_reference_step()

    # Keep the first fit's snapshot even after the caller advances the live optimizer.
    one_epoch = settings.model_copy(update={"epochs": 1})
    fit(model, data, one_epoch, optimizer=optimizer)
    check_against_next_reference_step()
    path = tmp_path / "checkpoints" / "best.pt"
    trainer.save_checkpoint(path)
    restored = Trainer(nn.Linear(1, 1), one_epoch)
    restored_optimizer = rmsprop(restored.model)
    assert restored.load_checkpoint(path, optimizer=restored_optimizer) == 1
    restored.fit(data, restored_optimizer)
    torch.testing.assert_close(restored.model.state_dict(), model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)


def test_checkpoint_rejects_incompatible_load_without_mutating_weights(tmp_path):
    split = TensorDataset(torch.ones(4, 1), torch.ones(4))
    data = DatasetSplits(split, split, split, 1, 0.0, 1.0)
    settings = TrainingSettings(device="cpu", epochs=1)
    trainer = Trainer(nn.Linear(1, 1), settings)
    path = tmp_path / "best.pt"
    with pytest.raises(RuntimeError, match="before saving"):
        trainer.save_checkpoint(path)
    trainer.fit(data)
    trainer.save_checkpoint(path)

    other = Trainer(nn.Linear(1, 1), settings)
    original = deepcopy(other.model.state_dict())
    wrong_type = torch.optim.RMSprop(other.model.parameters())
    reversed_group = torch.optim.Adam([other.model.bias, other.model.weight])
    for optimizer, message in [(wrong_type, "optimizer class"), (reversed_group, "parameter groups")]:
        with pytest.raises(ValueError, match=message):
            other.load_checkpoint(path, optimizer=optimizer)
        torch.testing.assert_close(other.model.state_dict(), original, rtol=0, atol=0)

    wrong_shape = Trainer(nn.Linear(2, 1), settings)
    original = deepcopy(wrong_shape.model.state_dict())
    with pytest.raises(RuntimeError, match="size mismatch"):
        wrong_shape.load_checkpoint(path)
    torch.testing.assert_close(wrong_shape.model.state_dict(), original, rtol=0, atol=0)
    with pytest.raises(ValueError, match="tracker"):
        other.load_checkpoint(run_id="run")
    with pytest.raises(ValueError, match="exactly one"):
        other.load_checkpoint(path, run_id="run")

    assert other.load_checkpoint(path) == 1
    assert not other.model.training
    torch.testing.assert_close(other.predict(split.tensors[0]), trainer.predict(split.tensors[0]))


def test_fit_rejects_optimizer_for_another_model():
    split = TensorDataset(torch.zeros(2, 1), torch.zeros(2))
    data = DatasetSplits(split, split, split, 1, 0.0, 1.0)
    trainer = Trainer(nn.Linear(1, 1), TrainingSettings(device="cpu", epochs=1))
    optimizer = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1)
    with pytest.raises(ValueError, match="model parameters"):
        trainer.fit(data, optimizer=optimizer)


def test_evaluate_weights_every_test_example_in_original_units():
    x = torch.zeros(3, 1)
    test = TensorDataset(x, torch.tensor([0.0, 0.0, 2.0]))
    data = DatasetSplits(test, test, test, 1, 42.0, 3.0)
    model = nn.Sequential(nn.Linear(1, 1), nn.Flatten(start_dim=0))
    nn.init.zeros_(model[0].bias)
    assert evaluate(model, data, batch_size=2) == pytest.approx(sqrt(12))


def test_training_settings_share_dotenv_with_models(tmp_path):
    env = tmp_path / ".env"
    env.write_text("MGS_KAN_FEATURES=8\nMGS_KAN_TRAIN_EPOCHS=3\n")
    assert ModelSettings(_env_file=env).features == 8
    assert TrainingSettings(_env_file=env).epochs == 3


@pytest.mark.parametrize("values", [{"epochs": 0}, {"batch_size": 0}, {"learning_rate": float("inf")}])
def test_training_settings_validate_parameters(values):
    with pytest.raises(ValidationError):
        TrainingSettings(**values)


def test_trainer_api_matches_function_calls():
    x = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    y = x.sum(dim=1)
    split = TensorDataset(x, y)
    data = DatasetSplits(split, split, split, 2, 10.0, 2.0)
    settings = TrainingSettings(device="cpu", seed=17, epochs=3, batch_size=2)
    model = nn.Linear(2, 1)
    reference = deepcopy(model)
    trainer = Trainer(model, settings)

    assert trainer.model is model
    assert trainer.fit(data) == fit(reference, data, settings)
    predictions = trainer.predict(x)
    assert predictions.shape == y.shape
    assert not predictions.requires_grad
    torch.testing.assert_close(predictions, reference(x).squeeze(-1), rtol=0, atol=0)
    assert trainer.evaluate(data) == evaluate(reference, data, batch_size=settings.batch_size)
    assert Trainer(model).evaluate(data) == evaluate(model, data)
    assert not model.training
