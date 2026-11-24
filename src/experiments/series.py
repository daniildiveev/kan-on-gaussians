"""A reproducible grid of model, fold and seed combinations."""

import json
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import ModelSettings, ResourceSettings, TrainingSettings
from src.data import CsvSpec, dataset_class
from src.models.factory import ModelName


@dataclass(frozen=True)
class Trial:
    model: str
    fold: int
    init_seed: int
    shuffle_seed: int

    @property
    def key(self) -> str:
        return f"{self.model}/{self.fold}/{self.init_seed}/{self.shuffle_seed}"


_Seed = Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]
_Fold = Annotated[int, Field(strict=True, ge=0, le=4)]


class ExperimentSeries(BaseModel):
    """Hold validated settings shared by all attempts in a named series."""

    model_config = ConfigDict(frozen=True)

    series_id: str
    dataset: str | CsvSpec
    models: tuple[ModelName, ...] = Field(min_length=1)
    feature_columns: tuple[str, ...] | None = None
    evaluation: Literal["test", "validation"] = "test"
    folds: tuple[_Fold, ...] = Field(min_length=1)
    seeds: tuple[tuple[_Seed, _Seed], ...] = Field(min_length=1)
    device: str = Field(pattern=r"^(cpu|cuda:(0|[1-9][0-9]*))$")
    training: TrainingSettings
    resources: ResourceSettings
    model_settings: ModelSettings

    @field_validator("series_id")
    @classmethod
    def _named_series(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("series_id must contain a name")
        return value

    @field_validator("models", "folds", "seeds")
    @classmethod
    def _unique_entries(cls, values: tuple) -> tuple:
        if len(values) != len(set(values)):
            raise ValueError("Models, folds and seed pairs must be unique")
        return values

    @model_validator(mode="after")
    def _supported_folds(self):
        if isinstance(self.dataset, str):
            reader = dataset_class(self.dataset)
            supported = reader.available_folds
            if self.feature_columns is not None:
                cols = self.feature_columns
                if not cols or len(set(cols)) != len(cols) or not set(cols).issubset(reader.columns):
                    raise ValueError(f"Invalid feature columns for {self.dataset}: {cols}")
            if not set(self.folds).issubset(supported):
                raise ValueError(f"{self.dataset} supports folds {supported}")
        elif self.feature_columns is not None:
            raise ValueError("Named feature selection requires a registered dataset")
        return self

    @property
    def key(self) -> str:
        return sha256(self.series_id.encode()).hexdigest()

    @property
    def size(self) -> int:
        return len(self.models) * len(self.folds) * len(self.seeds)

    def trials(self) -> tuple[Trial, ...]:
        return tuple(
            Trial(model, fold, *seeds)
            for fold, seeds, model in product(self.folds, self.seeds, self.models)
        )

    def protocol(self, identity: dict) -> str:
        settings = {
            **identity,
            **self.model_dump(include={"dataset", "feature_columns"}, exclude_none=True),
            "evaluation": self.evaluation,
            "models": sorted(self.models),
            "folds": sorted(self.folds),
            "seeds": sorted(self.seeds),
            "optimizer": "Adam",
            "model_settings": self.model_settings.model_dump(exclude={"features", "init_seed"}),
            "training": self.training.model_dump(exclude={"seed", "device"}),
            "cpu_threads": self.resources.cpu_threads,
            "backend": self.device.split(":")[0],
        }
        encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()
