"""Quality scores for red and white Vinho Verde using 11 chemical measurements."""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from src.data.base import Table, TabularDataset


class WineQualityDataset(TabularDataset):
    filename = "wine_quality.zip"
    url = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
    target = "quality"
    features = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
    ]

    columns = tuple(features)

    @classmethod
    def read(cls, path: Path) -> Table:
        tables = []
        with ZipFile(path) as archive:
            for color in ("red", "white"):
                with archive.open(f"winequality-{color}.csv") as source:
                    tables.append(pd.read_csv(source, sep=";", usecols=[*cls.features, cls.target]))
        data = pd.concat(tables, ignore_index=True)
        x = data[cls.features].to_numpy()
        return x, data[cls.target].to_numpy(), cls.group_ids(x)
