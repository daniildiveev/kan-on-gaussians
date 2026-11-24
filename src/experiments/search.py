"""Named validation grids. One resumable ExperimentRunner series per dataset and stage."""

from math import isqrt
from pathlib import Path
from time import monotonic
from typing import Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import ModelSettings, MonitorSettings, ResourceSettings, TrackingSettings, TrainingSettings
from src.data import dataset_class
from src.experiments.failures import failure_reason
from src.experiments.features import FeaturePolicy, benchmark_feature_policies
from src.experiments.runner import ExperimentRunner
from src.experiments.series import ExperimentSeries
from src.models.factory import ModelName
from src.monitoring.client import MonitorSession
from src.paths import DATA_DIR


class SearchStage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_-]+$")
    models: tuple[ModelName, ...]
    settings: ModelSettings

    def matched_settings(self, inputs: int) -> ModelSettings:
        """Find the nearest integer MLP width for this stage's MGS-KAN parameter count."""
        if not {"mlp", "mgs_kan"}.issubset(self.models):
            return self.settings
        pairs = inputs * (inputs - 1) // 2
        budget = (inputs + 1 + 3 * inputs * self.settings.single_kernels
                  + 6 * pairs * self.settings.pair_kernels)
        # Two hidden layers of width w have w² + (inputs+3)w + 1 parameters.
        linear = inputs + 3
        lower = max(1, (isqrt(linear**2 + 4 * (budget - 1)) - linear) // 2)
        width = min((lower, lower + 1), key=lambda w: abs(w*w + linear*w + 1 - budget))
        return self.settings.model_copy(update={"mlp_width": width})


AIRFOIL_STAGES = tuple(
    SearchStage(
        name=f"stage-{index}", models=("mlp", "mgs_kan"),
        settings=ModelSettings(mlp_width=width, single_kernels=single, pair_kernels=pair),
    )
    for index, (width, single, pair) in enumerate([(7, 3, 1), (10, 4, 2), (15, 7, 4)], start=1)
)
DEEP_STAGE = SearchStage(
    name="deep-128x4", models=("deep_mlp", "resnet"),
    settings=ModelSettings(deep_width=128, deep_depth=4),
)
FEATURE_STAGES = (
    *AIRFOIL_STAGES,
    SearchStage(name="deep-64x3", models=("deep_mlp",), settings=ModelSettings(deep_width=64, deep_depth=3)),
    SearchStage(name="resnet-64x2", models=("resnet",), settings=ModelSettings(deep_width=64, deep_depth=2)),
    DEEP_STAGE,
    SearchStage(name="deep-256x6", models=("deep_mlp", "resnet"), settings=ModelSettings(deep_width=256, deep_depth=6)),
)
BENCHMARKS = (
    "airfoil", "power_plant", "sarcos", "gas_turbine_co", "gas_turbine_nox",
    "superconductivity", "appliances", "concrete", "wine_quality",
)


class ValidationGrid(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    datasets: tuple[str, ...] = ("airfoil",)
    stages: tuple[SearchStage, ...] = AIRFOIL_STAGES[:2]
    feature_policies: dict[str, FeaturePolicy] = Field(default_factory=dict)
    match_mlp_budget: bool = False
    folds: tuple[int, ...] = (0,)
    seeds: tuple[tuple[int, int], ...] = ((11, 101), (22, 202), (33, 303))
    training: TrainingSettings = Field(default_factory=lambda: TrainingSettings(
        epochs=100, batch_size=64, learning_rate=0.001,
    ))

    @classmethod
    def preset(
        cls, profile: Literal["airfoil_first", "airfoil_all", "deep", "features"],
        *, datasets: tuple[str, ...] | None = None,
    ) -> Self:
        if datasets is not None and profile != "features":
            raise ValueError("Dataset selection applies to the features profile")
        if profile == "features":
            selected = BENCHMARKS if datasets is None else datasets
            policies = benchmark_feature_policies()
            if any(name not in policies for name in selected):
                raise ValueError(f"Choose benchmark datasets from {BENCHMARKS}")
            return cls(datasets=selected, stages=FEATURE_STAGES, match_mlp_budget=True,
                       feature_policies={name: policies[name] for name in selected})

        if profile == "airfoil_first":
            return cls()
        if profile == "airfoil_all":
            return cls(stages=AIRFOIL_STAGES)
        if profile == "deep":
            return cls(datasets=BENCHMARKS, stages=(DEEP_STAGE,))
        raise ValueError(f"Unknown validation profile: {profile}")

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        names = [stage.name for stage in self.stages]
        if not names or len(names) != len(set(names)):
            raise ValueError("Provide distinct stage names")
        if not self.datasets or len(self.datasets) != len(set(self.datasets)):
            raise ValueError("Provide distinct datasets")
        if not self.feature_policies.keys() <= set(self.datasets):
            raise ValueError("Feature policies must belong to selected datasets")
        for dataset in self.datasets:
            self.subsets(dataset)
            cls = dataset_class(dataset)
            if not set(self.folds).issubset(cls.available_folds):
                raise ValueError(f"{dataset} supports folds {cls.available_folds}")
            for stage in self.stages:
                ExperimentSeries(
                    series_id=stage.name, dataset=dataset, models=stage.models, folds=self.folds,
                    seeds=self.seeds, device="cpu", model_settings=stage.settings,
                    training=self.training, resources=ResourceSettings(), evaluation="validation",
                )
        return self

    def subsets(self, dataset: str) -> dict[str, tuple[str, ...] | None]:
        policy = self.feature_policies.get(dataset)
        return policy.subsets(dataset) if policy is not None else {"all": None}

    def plan(self):
        """Yield one resumable series for each dataset, feature set and model stage."""
        for dataset in self.datasets:
            for label, columns in self.subsets(dataset).items():
                for stage in self.stages:
                    settings = stage.settings
                    if self.match_mlp_budget:
                        settings = stage.matched_settings(len(columns or dataset_class(dataset).columns))
                    yield dataset, label, columns, stage, settings

    @property
    def size(self) -> int:
        return sum(len(self.subsets(d)) for d in self.datasets) * len(self.folds) * len(self.seeds) * sum(
            len(stage.models) for stage in self.stages
        )

    def run(
        self, *, version: str, device: str, tracking: TrackingSettings, resources: ResourceSettings,
        monitoring: MonitorSettings, root=DATA_DIR, download: bool = False,
    ) -> pd.DataFrame:
        started = monotonic()
        grid_id = f"{', '.join(self.datasets)} · {version}"
        total = self.size
        session = MonitorSession(monitoring, grid_id, total, device, grid_id=grid_id)
        try:
            session.start()
            runs = self._run_grid(session, version, device, tracking, resources, root, download)
            if len(runs) != total or runs["run_id"].nunique() != total:
                raise RuntimeError("Missing or duplicate training runs in validation grid")
        except KeyboardInterrupt:
            session.finish("interrupted")
            raise
        except Exception as error:
            session.finish("failed", failure_reason(error))
            raise
        session.finish("finished", elapsed_seconds=int(monotonic() - started))
        return runs

    def _run_grid(
        self, session: MonitorSession, version: str, device: str, tracking: TrackingSettings,
        resources: ResourceSettings, root: str | Path, download: bool,
    ) -> pd.DataFrame:
        rows = []
        runner = ExperimentRunner(training=self.training, resources=resources, tracking=tracking,
                                  monitoring=session.settings)
        for dataset, label, columns, stage, settings in self.plan():
            suffix = f"-features-{label}" if columns is not None else ""
            series = ExperimentSeries(
                series_id=f"{dataset}-validation-{stage.name}{suffix}-{version}",
                dataset=dataset, feature_columns=columns, models=stage.models, folds=self.folds,
                seeds=self.seeds, device=device, evaluation="validation", training=self.training,
                resources=resources, model_settings=settings,
            )
            session.begin_series(series.series_id, len(rows))
            summary = runner.run_series(
                series, session, root=root, download=download, completed_before=len(rows),
            )
            for result in summary.results:
                run = runner.tracker.client.get_run(result.run_id)
                rows.append({
                    "dataset": dataset, "feature_set": label,
                    "feature_columns": run.data.params.get("feature_columns"),
                    "n_features": int(run.data.params["n_features"]), "stage": stage.name, "model": result.trial.model,
                    "fold": result.trial.fold, "init_seed": result.trial.init_seed,
                    "shuffle_seed": result.trial.shuffle_seed, "best_epoch": result.best_epoch,
                    "val_rmse": result.best_val_rmse, "run_id": result.run_id,
                    "parameters": int(run.data.params["trainable_parameters"]),
                    "seconds": run.data.metrics["duration_seconds"],
                })
        return pd.DataFrame(rows)

    def compare(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Select within each family and outer fold, comparing the same seed pairs."""
        if len(rows) != self.size:
            raise ValueError("Selection requires a complete validation grid")
        summaries = []
        if "feature_set" not in rows and not self.feature_policies:
            rows = rows.assign(feature_set="all")
        expected_groups = {
            (d, fold, label, model) for d, label, _, stage, _ in self.plan()
            for fold in self.folds for model in stage.models
        }
        groups = rows.groupby(["dataset", "fold", "feature_set", "model"])
        if set(groups.groups) != expected_groups:
            raise ValueError("Selection requires the planned datasets, folds, feature sets and models")
        for (dataset, fold, label, model), group in groups:
            scores = group.pivot(index=["init_seed", "shuffle_seed"], columns="stage", values="val_rmse")
            expected = [s.name for s in self.stages if model in s.models]
            paired = set(scores.index) == set(self.seeds) and set(scores.columns) == set(expected)
            if not paired or scores.isna().any().any():
                raise ValueError("Selection requires all paired validation scores")
            stats = group.groupby("stage").agg(
                parameters=("parameters", "first"), mean_val_rmse=("val_rmse", "mean"),
                mean_seconds=("seconds", "mean"), latest_best_epoch=("best_epoch", "max"),
            ).sort_values("parameters", kind="stable")
            winner = stats.index[0]
            for candidate in stats.index[1:]:
                wins = (scores[candidate] < scores[winner]).sum()
                improves_mean = stats.at[candidate, "mean_val_rmse"] <= 0.99 * stats.at[winner, "mean_val_rmse"]
                if improves_mean and wins > len(scores) / 2:
                    winner = candidate
            stats["selected"] = stats.index == winner
            columns = self.subsets(dataset)[label] or dataset_class(dataset).columns
            summaries.append(stats.reset_index().assign(
                dataset=dataset, fold=fold, model=model, feature_set=label,
                feature_columns=", ".join(columns), n_features=len(columns),
            ))
        return pd.concat(summaries, ignore_index=True)

    @staticmethod
    def feature_reports(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Summarize variability across subsets after averaging paired seeds within each subset."""
        keys = ["dataset", "fold", "stage", "n_features"]
        per_subset = runs.groupby([*keys, "feature_set", "model"]).val_rmse.mean()
        distribution = per_subset.groupby([*keys, "model"]).agg(
            subsets="size", median_rmse="median", min_rmse="min", max_rmse="max",
        )
        baselines = runs[runs.model.isin(["mlp", "mgs_kan"])]
        pairs = baselines.pivot(index=[*keys, "feature_set", "init_seed", "shuffle_seed"],
                                columns="model", values="val_rmse")
        delta = (pairs["mlp"] - pairs["mgs_kan"]).groupby([*keys, "feature_set"]).mean()
        paired = delta.groupby(keys).agg(
            subsets="size", median_delta="median", min_delta="min", max_delta="max",
        )
        return distribution.reset_index(), paired.reset_index()
