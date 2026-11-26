"""Wire contract shared by the training host and standalone monitor."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


Seed = Annotated[int, Field(ge=0, le=2**64 - 1)]
Reason = Literal["cuda_oom", "disk_full", "write_error", "mlflow_unavailable",
                 "worker_exit", "series_error"]
Kind = Literal["started", "heartbeat", "finished", "paused", "interrupted", "failed"]


class TrialSnapshot(Snapshot):
    model: Literal["mlp", "mgs_kan"]
    fold: int = Field(ge=0, le=4)
    init_seed: Seed
    shuffle_seed: Seed


class DiskSample(Snapshot):
    path: str = Field(min_length=1, max_length=512)
    free_gib: float = Field(ge=0)
    threshold_gib: float = Field(gt=0)
    margin_gib: float = Field(gt=0)


class FirstResult(Snapshot):
    model: Literal["mlp", "mgs_kan", "deep_mlp", "resnet"]
    fold: int = Field(ge=0, le=4)
    run_id: str = Field(min_length=1, max_length=128)
    val_rmse: float = Field(ge=0)


class GridMilestone(Snapshot):
    # 25 and 75 remain valid for active sessions and queued notifications from older clients.
    percent: Literal[10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
    completed: int = Field(gt=0)
    series_id: str = Field(min_length=1, max_length=160)
    elapsed_seconds: int = Field(ge=0)


class MonitorEvent(Snapshot):
    session_id: UUID
    series_id: str = Field(min_length=1, max_length=160)
    grid_id: str | None = Field(default=None, min_length=1, max_length=160)
    first_result: FirstResult | None = None
    milestones: tuple[GridMilestone, ...] | None = Field(default=None, max_length=9)
    elapsed_seconds: int | None = Field(default=None, ge=0)
    sequence: int = Field(ge=1, le=2**63 - 1)
    kind: Kind
    total: int = Field(gt=0)
    completed: int | None = Field(default=None, ge=0)
    device: str = Field(max_length=64, pattern=r"^(cpu|cuda:[0-9]+)$")
    trial: TrialSnapshot | None = None
    run_id: str | None = Field(default=None, max_length=128)
    reason: Reason | None = None
    disk: DiskSample | None = None

    @property
    def terminal(self) -> bool:
        return self.kind not in ("started", "heartbeat")

    @model_validator(mode="after")
    def check_progress(self) -> Self:
        if self.completed is not None and self.completed > self.total:
            raise ValueError("Completed runs exceed the grid size")
        if self.grid_id is not None and self.kind == "finished":
            if self.completed != self.total or self.elapsed_seconds is None:
                raise ValueError("A finished grid requires all runs and elapsed time")
        return self
