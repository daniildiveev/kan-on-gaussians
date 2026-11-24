"""Predeclared feature subsets for validation-only model selection."""

from itertools import combinations

from pydantic import BaseModel, ConfigDict, Field

from src.data import dataset_class
from src.data.superconductivity import PROPERTIES


class FeaturePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exhaustive: bool = False
    groups: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    def subsets(self, dataset: str) -> dict[str, tuple[str, ...]]:
        columns = dataset_class(dataset).columns
        if not columns:
            raise ValueError(f"{dataset} has no named feature schema")
        subsets = {"all": columns}
        if self.exhaustive:
            if self.groups or dataset not in ("airfoil", "power_plant"):
                raise ValueError("Exhaustive search is reserved for Airfoil and Power Plant")
            for k in range(1, len(columns)):
                for indices in combinations(range(len(columns)), k):
                    label = "cols-" + "-".join(str(i + 1) for i in indices)
                    subsets[label] = tuple(columns[i] for i in indices)
        else:
            for label, names in self.groups.items():
                chosen = tuple(c for c in columns if c in names)
                if not label or label in subsets or not chosen or len(chosen) != len(names):
                    raise ValueError(f"Invalid feature group {label!r} for {dataset}: {names}")
                if chosen in subsets.values():
                    raise ValueError(f"Repeated feature subset in {dataset}: {label}")
                subsets[label] = chosen
        return subsets


def benchmark_feature_policies() -> dict[str, FeaturePolicy]:
    sarcos = dataset_class("sarcos").columns
    motion = {"position": sarcos[:7], "velocity": sarcos[7:14], "acceleration": sarcos[14:]}
    for a, b in combinations(tuple(motion), 2):
        motion[f"{a}-{b}"] = motion[a] + motion[b]
    turbine = FeaturePolicy(groups={
        "ambient": ("AT", "AP", "AH"),
        "operation": ("AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"),
    })
    chemistry = dataset_class("superconductivity").columns
    appliances = dataset_class("appliances").columns
    concrete = dataset_class("concrete").columns
    return {
        "airfoil": FeaturePolicy(exhaustive=True),
        "power_plant": FeaturePolicy(exhaustive=True),
        "sarcos": FeaturePolicy(groups=motion),
        "gas_turbine_co": turbine, "gas_turbine_nox": turbine,
        "superconductivity": FeaturePolicy(groups={
            prop: tuple(c for c in chemistry if c.endswith("_" + prop)) for prop in PROPERTIES
        }),
        "appliances": FeaturePolicy(groups={
            "room": appliances[:18], "weather": appliances[18:24], "clock": appliances[24:],
            "sensors": appliances[:24],
        }),
        "concrete": FeaturePolicy(groups={
            "recipe": concrete[:7], "binders": concrete[:3], "age": concrete[7:],
        }),
        "wine_quality": FeaturePolicy(groups={
            "acidity": ("fixed acidity", "volatile acidity", "citric acid", "pH"),
            "sulfur": ("free sulfur dioxide", "total sulfur dioxide", "sulphates"),
            "sugar-density-alcohol": ("residual sugar", "density", "alcohol"),
        }),
    }
