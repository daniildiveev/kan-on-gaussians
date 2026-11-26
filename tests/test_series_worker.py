import json
import os
import subprocess
import sys
from hashlib import sha256

import pytest
import torch
from pydantic import ValidationError

from src.config import ResourceSettings


def test_resource_limits_from_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MGS_KAN_SERIES_CPU_THREADS=3\nMGS_KAN_SERIES_GPU_MEMORY_FRACTION=0.5\n")
    monkeypatch.delenv("MGS_KAN_SERIES_CPU_THREADS", raising=False)
    monkeypatch.delenv("MGS_KAN_SERIES_GPU_MEMORY_FRACTION", raising=False)
    assert ResourceSettings().cpu_threads == 2
    assert ResourceSettings().gpu_memory_fraction == 0.25
    settings = ResourceSettings(_env_file=env)
    assert (settings.cpu_threads, settings.gpu_memory_fraction) == (3, 0.5)
    assert settings.max_parallel_trials == 1 and settings.total_gpu_budget_mib is None
    assert ResourceSettings(max_parallel_trials=4).total_gpu_budget_mib == 4096
    assert ResourceSettings(gpu_memory_fraction=1).gpu_memory_fraction == 1
    for invalid in [
        {"max_parallel_trials": 0}, {"max_parallel_trials": 5}, {"gpu_budget_mib": 0},
        {"cpu_threads": 0}, {"cpu_threads": 1.5}, {"gpu_memory_fraction": 0},
        {"gpu_memory_fraction": 1.1}, {"gpu_memory_fraction": float("nan")},
    ]:
        with pytest.raises(ValidationError):
            ResourceSettings(**invalid)


@pytest.fixture
def request_data(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    (tmp_path / "table.csv").write_text("x,y\n" + "".join(f"{i},{i * i}\n" for i in range(20)))
    return {
        "action": "probe", "series_key": "demo-key", "device": "cpu", "root": str(tmp_path),
        "dataset": {"filename": "table.csv", "target": "y"}, "download": False, "fold": 1,
        "resources": {"cpu_threads": 2, "gpu_memory_fraction": 0.25},
        "training": {"epochs": 1}, "model_settings": {},
        "tracking": {"tracking_uri": (tmp_path / "runs").as_uri(), "experiment_name": "series"},
        "model": "mlp", "init_seed": 123, "shuffle_seed": 456,
        "tags": {"mgs_kan.series": "demo", "mgs_kan.protocol": "p", "mgs_kan.trial": "t"},
    }


@pytest.fixture
def worker(monkeypatch):
    pytest.importorskip("mlflow")
    from src.experiments import worker

    # Inter-op pools may only be initialized once in a process. Each real worker exits.
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda threads: None)
    old_threads = torch.get_num_threads()
    yield worker
    torch.set_num_threads(old_threads)


def test_probe_and_trial_detect_changed_data(worker, request_data, tmp_path, monkeypatch):
    def unexpected_cuda():
        pytest.fail("CPU probes must avoid CUDA initialization")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda)
    before = worker.TrialWorker(request_data).execute()["identity"]
    path = tmp_path / "table.csv"
    assert before["dataset_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert len(before["source_sha256"]) == 64
    assert before["device_model"] and before["device_model"] != "cpu"
    with path.open("a") as output:
        output.write("21,441\n")
    with pytest.raises(ValueError, match="dataset_sha256"):
        worker.TrialWorker({**request_data, "action": "trial", "expected": before}).execute()


def test_cpu_trial_rejects_different_processor(worker, request_data, tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    real_read = worker.Path.read_text

    def read(path, **kwargs):
        return real_read(cpuinfo if str(path) == "/proc/cpuinfo" else path, **kwargs)

    monkeypatch.setattr(worker.sys, "platform", "linux")
    monkeypatch.setattr(worker.Path, "read_text", read)
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Example CPU A\n")
    expected = worker.TrialWorker(request_data).execute()["identity"]
    assert expected["device_model"] == "Example CPU A"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Example CPU B\n")
    with pytest.raises(ValueError, match="device_model"):
        worker.TrialWorker({**request_data, "action": "trial", "expected": expected}).execute()


def test_visible_cuda_limit_applied_before_model(worker, request_data, monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda dev: events.append(("device", str(dev))))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev: "test GPU")
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, dev: events.append(("limit", fraction, str(dev))))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda dev: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev: 2048)
    request_data["device"] = "cuda:1"
    with pytest.raises(ValueError, match="visible"):
        worker.TrialWorker({**request_data, "device": "cuda:2"}).execute()
    expected = worker.TrialWorker(request_data).execute()["identity"]
    assert expected["device_model"] == "test GPU"
    events.clear()

    def stop_at_model(name, settings):
        events.append(("model", name, settings.init_seed, settings.features))
        raise RuntimeError("stop before real CUDA work")

    monkeypatch.setattr(worker, "make_model", stop_at_model)
    with pytest.raises(RuntimeError, match="stop before real CUDA"):
        worker.TrialWorker({**request_data, "action": "trial", "expected": expected}).execute()
    assert events == [("device", "cuda:1"), ("limit", 0.25, "cuda:1"), ("model", "mlp", 123, 1)]


@pytest.mark.parametrize("failure_point", ["_fit_epochs", "evaluate"])
def test_failed_attempt_records_timing_and_peak(worker, request_data, monkeypatch, failure_point):
    from mlflow import MlflowClient

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    # Simulate the allocator only. The model, optimizer and MLflow client are real.
    request_data["device"] = "cuda:1"
    monkeypatch.setattr(worker.TrialWorker, "_configure_device", lambda *_: "Example CUDA GPU")

    def place_on_cpu(trainer, settings):
        trainer.device = torch.device("cpu")

    monkeypatch.setattr(worker.Trainer, "_place_model", place_on_cpu)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: {"cuda:1": 1234567}[device])
    expected = worker.TrialWorker(request_data).execute()["identity"]

    def exhaust_memory(*args):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(worker.Trainer, failure_point, exhaust_memory)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="out of memory"):
        worker.TrialWorker({**request_data, "action": "trial", "expected": expected}).execute()
    client = MlflowClient(tracking_uri=request_data["tracking"]["tracking_uri"])
    experiment = client.get_experiment_by_name("series")
    run, = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FAILED"
    assert run.data.tags.get("mgs_kan.evaluated") != "true"
    assert "out of memory" in run.data.tags["mgs_kan.failure"]
    assert run.data.tags["mgs_kan.cpu_threads"] == "2"
    assert run.data.params["shuffle_seed"] == "456"
    assert run.data.metrics["duration_seconds"] > 0
    assert run.data.metrics["cuda_peak_allocated_bytes"] == 1234567


@pytest.mark.skipif(os.name != "posix", reason="POSIX workers detect reparenting")
def test_worker_exits_when_kernel_has_gone(request_data, tmp_path):
    pytest.importorskip("mlflow")
    result = tmp_path / "result.json"
    # The recorded parent no longer matches the process that owns this worker.
    process = subprocess.run(
        [sys.executable, "-m", "src.experiments.worker", str(result)],
        input=json.dumps({**request_data, "parent_pid": -1}) + "\n",
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 1
    assert "Traceback" not in process.stderr
    assert not result.exists()


def test_gpu_budget_subtracts_context_and_splits_capacity(worker, request_data, monkeypatch):
    from types import SimpleNamespace

    request_data.update(device="cuda:0", resources={"max_parallel_trials": 4, "gpu_budget_mib": 4096})
    device = torch.device("cuda:0")
    limits = []
    monkeypatch.setattr(worker, "_resolve_device", lambda _: device)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: object())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=24 * 2**30))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, dev: limits.append(fraction))
    monkeypatch.setattr(worker, "gpu_memory_by_pid", lambda pids: {os.getpid(): 256})
    worker.TrialWorker(request_data)._configure_device()
    assert limits == [(1024 - 256) / (24 * 1024)]
    monkeypatch.setattr(worker, "gpu_memory_by_pid", lambda pids: {})
    with pytest.raises(RuntimeError, match="worker PID"):
        worker.TrialWorker(request_data)._configure_device()
