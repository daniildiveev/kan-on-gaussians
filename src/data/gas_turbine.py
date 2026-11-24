"""Hourly gas turbine emissions, evaluated on the last two calendar years."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.data.base import Table, TabularDataset


class GasTurbineDataset(TabularDataset):
    filename = "gas_turbine.zip"
    url = "https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip"
    features = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
    columns = tuple(features)
    available_folds = (0,)
    target: str

    @classmethod
    def read(cls, path: Path) -> Table:
        tables = []
        years = []
        with ZipFile(path) as archive:
            for year in range(2011, 2016):
                with archive.open(f"gt_{year}.csv") as source:
                    table = pd.read_csv(source, usecols=[*cls.features, cls.target])
                tables.append(table)
                years.append(np.full(len(table), year))
        data = pd.concat(tables, ignore_index=True)
        return data[cls.features].to_numpy(), data[cls.target].to_numpy(), np.concatenate(years)

    @classmethod
    def split_indices(cls, years: np.ndarray, fold: int) -> dict[str, np.ndarray]:
        return {
            "train": np.flatnonzero(years < 2013),
            "val": np.flatnonzero(years == 2013),
            "test": np.flatnonzero(years >= 2014),
        }


class GasTurbineCODataset(GasTurbineDataset):
    target = "CO"


class GasTurbineNOxDataset(GasTurbineDataset):
    target = "NOX"
