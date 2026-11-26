"""Match the effective training settings recorded by legacy and current validation runs."""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.config import ModelSettings
from src.data import DatasetSplits
from src.models import make_model, parameter_count

if TYPE_CHECKING:
    from src.experiments.series import ExperimentSeries, Trial


def optimizer_params(optimizer: torch.optim.Optimizer) -> dict[str, str]:
    params = {}
    for index, group in enumerate(optimizer.param_groups):
        for name, value in group.items():
            if name in ("params", "param_names"):
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().tolist()
            params[f"optimizer.{name}.{index}"] = json.dumps(value, default=str)
    return params


def _active_model_params(model: str, settings: ModelSettings) -> dict:
    # Earlier grids logged all families' settings together, even those unused by this model.
    if model == "mlp":
        fields = {"mlp_width"}
    elif model == "mgs_kan":
        fields = {name for name in ModelSettings.model_fields if name.startswith(("single_", "pair_"))}
    else:
        fields = {"deep_width", "deep_depth"}
    options = settings.model_dump(include=fields | {"features", "init_seed"})
    return {f"model.{name}": str(value) for name, value in options.items()}


@dataclass(frozen=True)
class ValidationRecipe:
    params: dict
    optimizer: dict
    columns: list[str]
    backend: str

    @classmethod
    def for_trial(cls, series: "ExperimentSeries", trial: "Trial", data: DatasetSplits):
        settings = series.model_settings.model_copy(update={"features": data.n_features, "init_seed": trial.init_seed})
        model = make_model(trial.model, settings)
        adam = torch.optim.Adam(model.parameters(), lr=series.training.learning_rate)
        params = {
            "dataset": data.source_file, "dataset_sha256": data.source_sha256, "target_column": data.target_column,
            "fold": str(trial.fold), "n_features": str(data.n_features), "epochs": str(series.training.epochs),
            "batch_size": str(series.training.batch_size), "optimizer": "Adam", "model": trial.model,
            "seed": str(trial.shuffle_seed), "shuffle_seed": str(trial.shuffle_seed),
            "trainable_parameters": str(parameter_count(model)),
            **_active_model_params(trial.model, settings),
        }
        columns = list(data.selected_features or data.columns)
        return cls(params, optimizer_params(adam), columns, series.device.split(":")[0])

    def matches(self, recorded: dict) -> bool:
        if any(recorded.get(k) != v for k, v in self.params.items()):
            return False
        if recorded.get("device", "").split(":")[0] != self.backend:
            return False
        try:
            columns = json.loads(recorded.get("feature_columns", "null"))
        except (ValueError, TypeError):
            return False
        optimizer = {k: v for k, v in recorded.items() if k.startswith("optimizer.")}
        return self.columns == columns and self.optimizer == optimizer
