"""Appliance electricity use in Wh during each ten-minute interval."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.data.base import Table, TabularDataset


class AppliancesDataset(TabularDataset):
    filename = "appliances.zip"
    url = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
    target = "Appliances"
    available_folds = (0,)
    room_features = [name for i in range(1, 10) for name in (f"T{i}", f"RH_{i}")]
    weather_features = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]

    columns = (*room_features, *weather_features, "time_sin", "time_cos")

    @classmethod
    def read(cls, path: Path) -> Table:
        with ZipFile(path) as archive, archive.open("energydata_complete.csv") as source:
            data = pd.read_csv(source)
        dates = pd.to_datetime(data["date"], format="%Y-%m-%d %H:%M:%S")
        if dates.isna().any() or dates.duplicated().any():
            raise ValueError("Expected distinct timestamps in the appliances data")
        minutes = dates.dt.hour * 60 + dates.dt.minute
        phase = 2 * np.pi * minutes / (24 * 60)
        sensors = data[cls.room_features + cls.weather_features]
        x = np.column_stack((sensors, np.sin(phase), np.cos(phase)))
        return x, data[cls.target].to_numpy(), dates.to_numpy()

    @classmethod
    def split_indices(cls, dates: np.ndarray, fold: int) -> dict[str, np.ndarray]:
        ordered = np.argsort(dates)
        train, val, test = np.split(ordered, [len(dates) * 70 // 100, len(dates) * 85 // 100])
        return {"train": train, "val": val, "test": test}
