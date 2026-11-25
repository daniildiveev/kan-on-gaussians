"""Run one series probe or training attempt in a disposable Python process."""

import json
import os
import platform
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from threading import Thread
from time import monotonic, sleep

import torch

from src.config import ModelSettings, ResourceSettings, TrackingSettings, TrainingSettings
from src.data import CsvSpec, DatasetSplits, load_dataset
from src.models import make_model
from src.tracking import MlflowTracker
from src.tracking.provenance import SourceVersion
from src.training import Trainer
from src.training.trainer import _resolve_device
from src.experiments.failures import failure_reason


def _cpu_model() -> str:
    if sys.platform == "linux":
        with suppress(OSError):
            info = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            fields = {}
            for line in info.splitlines():
                key, _, value = line.partition(":")
                fields.setdefault(key.strip(), value.strip())
            for key in ("model name", "Hardware"):
                if fields.get(key):
                    return fields[key]
            if fields.get("CPU implementer") and fields.get("CPU part"):
                return f"{fields['CPU implementer']} / {fields['CPU part']}"
    if sys.platform == "darwin":
        with suppress(OSError, subprocess.SubprocessError):
            name = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if name:
                return name
    return platform.processor() or platform.machine() or "unknown CPU"


class TrialWorker:
    """Apply one request's limits and run its probe or training attempt."""

    def __init__(self, request: dict):
        self.request = request
        self.resources = ResourceSettings(**request["resources"])
        self.device = request["device"]
        self.tracking = TrackingSettings(**request["tracking"])
        self.run_id = None

    def _configure_device(self) -> str:
        torch.set_num_threads(self.resources.cpu_threads)
        torch.set_num_interop_threads(self.resources.cpu_threads)
        if self.device == "cpu":
            return _cpu_model()
        if not self.device.startswith("cuda:") or not self.device[5:].isdecimal():
            raise ValueError("Choose cpu or an explicit visible cuda:N device")
        device = _resolve_device(self.device)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(self.resources.gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.get_device_name(device)

    def _log_resources(self, tracker: MlflowTracker, run_id: str, started: float) -> None:
        client = tracker.client
        client.log_metric(run_id, "duration_seconds", monotonic() - started, synchronous=True)
        if self.device != "cpu":
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
            client.log_metric(run_id, "cuda_peak_allocated_bytes", peak_bytes, synchronous=True)

    def execute(self) -> dict:
        started = monotonic()
        device_model = self._configure_device()
        request = self.request
        dataset = request["dataset"]
        columns = request.get("feature_columns")
        data = load_dataset(
            CsvSpec(**dataset) if isinstance(dataset, dict) else dataset,
            root=request["root"], fold=request["fold"], download=request["download"],
            feature_columns=tuple(columns) if columns is not None else None,
        )
        identity = {
            "dataset_sha256": data.source_sha256,
            "device_model": device_model,
            "source_sha256": SourceVersion().tags()["mgs_kan.source_sha256"],
        }
        if request["action"] == "probe":
            tracker = MlflowTracker(self.tracking)
            history = tracker.completed_trials(request["series_key"], evaluation=request.get("evaluation", "test"))
            return {"identity": identity, **history}
        if request["action"] != "trial":
            raise ValueError(f"Unknown series action {request['action']!r}")
        for key, value in identity.items():
            if value != request["expected"][key]:
                raise ValueError(f"Series {key} changed since the probe")
        return self._train(data, device_model, started)

    def _train(self, data: DatasetSplits, device_model: str, started: float) -> dict:
        request = self.request
        model_options = request["model_settings"] | {
            "features": data.n_features, "init_seed": request["init_seed"],
        }
        model = make_model(request["model"], ModelSettings(**model_options))
        training_options = request["training"] | {"device": self.device, "seed": request["shuffle_seed"]}
        tags = {
            **request["tags"], "mgs_kan.device": self.device, "mgs_kan.device_model": device_model,
            "mgs_kan.cpu_threads": str(self.resources.cpu_threads),
            "mgs_kan.gpu_memory_fraction": str(self.resources.gpu_memory_fraction),
        }
        tracker = MlflowTracker(self.tracking, run_tags=tags)
        trainer = Trainer(model, TrainingSettings(**training_options), tracker=tracker)
        client = tracker.client
        evaluated = False
        try:
            result = trainer.fit(data)
            score = result.validation_rmse[result.best_epoch - 1]
            if request.get("evaluation", "test") == "validation":
                client.log_metric(result.run_id, "best_val_rmse", score, synchronous=True)
            else:
                trainer.evaluate(data)
            evaluated = True
            self._log_resources(tracker, result.run_id, started)
            client.set_tag(result.run_id, "mgs_kan.evaluated", "true", synchronous=True)
        except (Exception, KeyboardInterrupt) as error:
            run_id = tracker.last_run_id
            self.run_id = run_id
            if run_id is None:
                raise
            # Preserve the original error even if the GPU or tracking server has gone away.
            if not evaluated:
                with suppress(Exception):
                    self._log_resources(tracker, run_id, started)
            with suppress(Exception):
                client.set_tag(run_id, "mgs_kan.failure", f"{type(error).__name__}: {error}"[:1000],
                               synchronous=True)
            with suppress(Exception):
                status = "KILLED" if isinstance(error, KeyboardInterrupt) else "FAILED"
                client.set_terminated(run_id, status=status)
            raise
        return {"run_id": result.run_id, "val_rmse": score}


def _watch_parent(parent_pid: int) -> None:
    while os.getppid() == parent_pid:
        sleep(0.5)
    # A dead notebook cannot deliver SIGINT or reap the worker.
    os._exit(1)


def main() -> None:
    result_path = Path(sys.argv[1])
    request = json.loads(sys.stdin.readline())
    if os.name == "posix":
        Thread(target=_watch_parent, args=(request["parent_pid"],), daemon=True).start()
    worker = TrialWorker(request)
    try:
        result = worker.execute()
    except Exception as error:
        with suppress(OSError):
            result_path.write_text(json.dumps({"reason": failure_reason(error), "run_id": worker.run_id}),
                                   encoding="utf-8")
        raise
    result_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
