"""UCI Combined Cycle Power Plant, one copy of the 9,568 observations."""

import numpy as np

from src.data.base import Table
from src.data.csv_dataset import CsvDataset


class PowerPlantDataset(CsvDataset):
    filename = "power_plant.csv"
    url = "https://archive.ics.uci.edu/static/public/294/data.csv"
    columns = ("AT", "V", "AP", "RH")
    target = "PE"

    @classmethod
    def prepare_columns(cls, header: list[str], values: np.ndarray) -> Table:
        if header != ["AT", "V", "AP", "RH", "PE"]:
            raise ValueError("Power Plant requires AT, V, AP, RH and PE")
        x = values[:, :4]
        return x, values[:, 4], cls.group_ids(x)
