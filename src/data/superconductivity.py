"""Critical temperature, evaluated on held-out chemical compositions."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.data.base import Table, TabularDataset


PROPERTIES = ("atomic_mass", "fie", "atomic_radius", "Density", "ElectronAffinity",
              "FusionHeat", "ThermalConductivity", "Valence")
STATISTICS = ("mean", "wtd_mean", "gmean", "wtd_gmean", "entropy", "wtd_entropy",
              "range", "wtd_range", "std", "wtd_std")


class SuperconductivityDataset(TabularDataset):
    filename = "superconductivity.zip"
    url = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
    columns = ("number_of_elements", *(f"{stat}_{prop}" for prop in PROPERTIES for stat in STATISTICS))
    target = "critical_temp"

    @classmethod
    def read(cls, path: Path) -> Table:
        with ZipFile(path) as archive:
            with archive.open("train.csv") as source:
                data = pd.read_csv(source)
            with archive.open("unique_m.csv") as source:
                formulas = pd.read_csv(source)
        y = data.pop(cls.target).to_numpy()
        if data.shape[1] != 81 or not np.array_equal(y, formulas.pop(cls.target).to_numpy()):
            raise ValueError("Expected 81 features and row-aligned composition and target files")
        amounts = formulas.drop(columns="material").to_numpy(dtype=float)
        totals = amounts.sum(axis=1, keepdims=True)
        if not np.isfinite(amounts).all() or np.any(totals <= 0):
            raise ValueError("Invalid element amounts in unique_m.csv")
        fractions = np.round(amounts / totals, decimals=12)
        return data.loc[:, list(cls.columns)].to_numpy(), y, cls.group_ids(fractions)
