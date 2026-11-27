"""Hourly gas turbine emissions, evaluated on the last two calendar years."""

from dataclasses import replace
from pathlib import Path
from typing import Self
from zipfile import ZipFile

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset

from src.config import GAS_TURBINE_WINDOW_SIZE
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


class GasTurbineWindowDataset(GasTurbineDataset):
    """Six ordered observations from one year, labelled at the final observation."""

    input_protocol = "within-year-window6-stride1-endpoint"

    @classmethod
    def _from_arrays(cls, x, y, groups, fold, **source) -> Self:
        data = super()._from_arrays(x, y, groups, fold, **source)
        partitions = {}
        for name, indices in cls.split_indices(groups, fold).items():
            features, targets = getattr(data, name).tensors
            # read() emits each year's rows consecutively, in source-file order.
            years = groups[indices]
            length = GAS_TURBINE_WINDOW_SIZE
            starts = np.arange(max(0, len(years) - length + 1))
            ends = starts + length - 1
            within_year = years[starts] == years[ends]
            if not within_year.any():
                raise ValueError(f"No complete within-year windows in {name}")
            windows = features.unfold(0, length, 1).transpose(1, 2)
            partitions[name] = TensorDataset(windows[starts[within_year]], targets[ends[within_year]])
        return replace(data, **partitions)

    def test_endpoints(self) -> TensorDataset:
        """Present the same held-out examples to a single-observation model."""
        windows, targets = self.test.tensors
        return TensorDataset(windows[:, -1], targets)


class GasTurbineCOWindows(GasTurbineWindowDataset):
    target = "CO"


class GasTurbineNOxWindows(GasTurbineWindowDataset):
    target = "NOX"
