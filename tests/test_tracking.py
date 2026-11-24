import json
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
import torch

from config import ModelSettings, TrackingSettings, TrainingSettings
from data import CsvSpec, load_dataset
from models import make_model
from training import Trainer


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from tracking import MlflowTracker

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "runs").as_uri()
    env = tmp_path / ".env"
    env.write_text(
        f"MLFLOW_TRACKING_URI={uri}\nMLFLOW_EXPERIMENT_NAME=regression\n"
        "MLFLOW_CHECKPOINT_FILE=winner.pt\n"
    )
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_CHECKPOINT_FILE"):
        monkeypatch.delenv(name, raising=False)
    return MlflowTracker(TrackingSettings(_env_file=env))


@pytest.fixture
def data(tmp_path):
    x = np.arange(60)
    np.savetxt(tmp_path / "small.csv", np.column_stack((x, x % 7)),
               delimiter=",", header="x,y", comments="")
    return load_dataset(CsvSpec("small.csv", "y"), root=tmp_path, fold=2)


def test_tracked_notebook_runs(tracker, data, tmp_path):
    settings = TrainingSettings(device="cpu", epochs=3, batch_size=3)
    model = make_model("mlp", ModelSettings(features=1, mlp_width=5, init_seed=17))
    trainer = Trainer(model, settings, tracker=tracker)
    optimizer = torch.optim.RMSprop([
        {"params": model[0].parameters(), "lr": 0.01, "momentum": 0.25},
        {"params": model[2:].parameters(), "lr": 0.02, "weight_decay": 0.03},
    ])
    result = trainer.fit(data, optimizer)
    client = tracker.client
    run = client.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params.items() >= {
        "model": "mlp", "model.mlp_width": "5", "dataset": "small.csv", "fold": "2",
        "dataset_sha256": sha256((tmp_path / "small.csv").read_bytes()).hexdigest(),
        "target_column": "y", "optimizer": "RMSprop", "optimizer.lr.0": "0.01",
        "optimizer.lr.1": "0.02", "device": "cpu", "seed": str(settings.seed),
        "optimizer.momentum.0": "0.25", "optimizer.weight_decay.1": "0.03",
        "model.init_seed": "17", "shuffle_seed": str(settings.seed),
    }.items()
    assert set(run.data.metrics) == {"train_mse", "val_rmse", "val_r2", "best_epoch"}
    assert run.data.metrics["best_epoch"] == result.best_epoch

    actual = data.inverse_target(data.val.tensors[1].double())
    variance = (actual - actual.mean()).square().mean().item()
    expected = {
        "train_mse": result.train_mse,
        "val_rmse": result.validation_rmse,
        "val_r2": [1 - rmse**2 / variance for rmse in result.validation_rmse],
    }
    for name, values in expected.items():
        history = sorted(client.get_metric_history(result.run_id, name), key=lambda m: m.step)
        assert [metric.step for metric in history] == [1, 2, 3]
        assert [metric.value for metric in history] == pytest.approx(values)

    second = trainer.fit(data)
    assert second.run_id != result.run_id
    trainer.evaluate(replace(data, test=data.val))
    assert "test_rmse" not in client.get_run(second.run_id).data.metrics
    rmse = trainer.evaluate(data)
    predictions = trainer.predict(data.test.tensors[0]).double()
    targets = data.test.tensors[1].double()
    r2 = 1 - (predictions - targets).square().sum() / (targets - targets.mean()).square().sum()
    scored = client.get_run(second.run_id)
    assert scored.data.params["optimizer"] == "Adam"
    assert json.loads(scored.data.params["optimizer.betas.0"]) == [0.9, 0.999]
    assert scored.data.metrics["test_rmse"] == rmse
    assert scored.data.metrics["test_r2"] == pytest.approx(r2.item())
    assert "test_rmse" not in client.get_run(result.run_id).data.metrics


def test_replay_from_logged_seeds_and_optimizer_options(tracker, data):
    settings = TrainingSettings(seed=85, epochs=2, batch_size=4)
    model = make_model("mlp", ModelSettings(features=1, init_seed=19))
    trainer = Trainer(model, settings, tracker=tracker)
    result = trainer.fit(data, torch.optim.Adam(model.parameters(), betas=(0.8, 0.95)))
    run = tracker.client.get_run(result.run_id)
    params = run.data.params
    assert len(run.data.tags["mgs_kan.source_sha256"]) == 64

    replay = make_model(params["model"], ModelSettings(features=1, init_seed=int(params["model.init_seed"])))
    options = {
        key.split(".")[1]: json.loads(value)
        for key, value in params.items() if key.startswith("optimizer.") and key.endswith(".0")
    }
    replay_settings = TrainingSettings(
        seed=int(params["shuffle_seed"]), epochs=int(params["epochs"]), batch_size=int(params["batch_size"])
    )
    actual = Trainer(replay, replay_settings).fit(data, torch.optim.Adam(replay.parameters(), **options))
    assert actual.train_mse == result.train_mse
    assert actual.validation_rmse == result.validation_rmse


def test_source_digest_without_git(tmp_path):
    pytest.importorskip("mlflow")
    from tracking.provenance import SourceVersion

    config = tmp_path / "config.py"
    config.write_text("epochs = 2\n")
    package = tmp_path / "models"
    package.mkdir()
    source = package / "mlp.py"
    source.write_text("width = 4\n")
    version = SourceVersion(tmp_path)
    first = version.tags()
    assert set(first) == {"mgs_kan.source_sha256"}
    (tmp_path / "other.py").write_text("x = 1\n")
    assert version.tags() == first
    source.write_text("width = 8\n")
    second = version.tags()
    assert second != first
    config.write_text("epochs = 3\n")
    assert version.tags() != second


def test_restore_checkpoint_from_an_earlier_run(tracker, data, tmp_path):
    from tracking import MlflowTracker

    settings = TrainingSettings(device="cpu", epochs=3)
    trainer = Trainer(torch.nn.Linear(1, 1), settings, tracker=tracker)
    optimizer = torch.optim.Adam(trainer.model.parameters())
    first = trainer.fit(data, optimizer)
    x, _ = data.test.tensors
    expected = trainer.predict(x)
    local_path = tmp_path / "local.pt"
    trainer.save_checkpoint(local_path)
    next_result = trainer.fit(data, optimizer)
    next_predictions = trainer.predict(x)

    assert [item.path for item in tracker.client.list_artifacts(first.run_id)] == ["winner.pt"]
    updated = tracker.settings.model_copy(update={"checkpoint_file": "next.pt"})
    restored = Trainer(torch.nn.Linear(1, 1), settings, tracker=MlflowTracker(updated))
    loaded_optimizer = torch.optim.Adam(restored.model.parameters(), lr=0.7)
    assert restored.load_checkpoint(run_id=first.run_id, optimizer=loaded_optimizer) == first.best_epoch
    torch.testing.assert_close(restored.predict(x), expected, rtol=0, atol=0)
    from_run = tmp_path / "from-run.pt"
    restored.save_checkpoint(from_run)
    assert from_run.read_bytes() == local_path.read_bytes()
    assert restored.fit(data, loaded_optimizer).train_mse == next_result.train_mse
    torch.testing.assert_close(restored.predict(x), next_predictions, rtol=0, atol=0)


def test_failed_fit_closes_run_and_keeps_completed_epochs(tracker, data, monkeypatch):
    trainer = Trainer(
        make_model("mlp", ModelSettings(features=1)),
        TrainingSettings(device="cpu", epochs=2),
        tracker=tracker,
    )
    train_epoch = trainer._train_epoch

    def fail_next_epoch(*args):
        monkeypatch.setattr(trainer, "_train_epoch", fail)
        return train_epoch(*args)

    def fail(*args):
        raise ValueError("training interrupted")

    monkeypatch.setattr(trainer, "_train_epoch", fail_next_epoch)
    with pytest.raises(ValueError, match="training interrupted"):
        trainer.fit(data)
    experiment = tracker.client.get_experiment_by_name("regression")
    run, = tracker.client.search_runs([experiment.experiment_id])
    assert run.info.status == "FAILED"
    assert "val_rmse" in run.data.metrics
    assert "best_epoch" not in run.data.metrics


@pytest.mark.parametrize(("rows", "predicted", "expected"), [(1, 0, None), (3, 0, 1), (3, 1, 0)])
def test_r2_for_constant_targets(tracker, data, rows, predicted, expected):
    split = torch.utils.data.TensorDataset(torch.ones(rows, 1), torch.zeros(rows))
    constant = replace(data, train=split, val=split, test=split)
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(predicted)
    trainer = Trainer(model, TrainingSettings(device="cpu", epochs=1), tracker=tracker)
    result = trainer.fit(constant, torch.optim.SGD(model.parameters(), lr=0.0))
    trainer.evaluate(constant)
    metrics = tracker.client.get_run(result.run_id).data.metrics
    assert metrics.get("val_r2") == expected
    assert metrics.get("test_r2") == expected
