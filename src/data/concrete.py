"""UCI Concrete Compressive Strength, following the archived mixture protocol."""

import numpy as np

from src.data.base import Table
from src.data.csv_dataset import CsvDataset


class ConcreteDataset(CsvDataset):
    filename = "concrete_compressive_strength.csv"
    url = "https://archive.ics.uci.edu/static/public/165/data.csv"
    HEADER = [
        "Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer",
        "Coarse Aggregate", "Fine Aggregate", "Age", "Concrete compressive strength",
    ]

    @classmethod
    def prepare_columns(cls, header: list[str], values: np.ndarray) -> Table:
        if header != cls.HEADER or values.shape != (1030, 9):
            raise ValueError(
                "Concrete requires the UCI header and 1030 rows with eight features and a target"
            )

        ingredients = values[:, :-2]
        age = values[:, -2]
        strength = values[:, -1]
        if np.any(age <= 0) or np.any(strength <= 0):
            raise ValueError("Concrete age and compressive strength must be positive")

        features = values[:, :-1].copy()
        features[:, -1] = np.log1p(age)
        _, recipe_groups = np.unique(ingredients, axis=0, return_inverse=True)
        return features, strength, recipe_groups.reshape(-1)
