"""GPML inverse dynamics benchmark for the torque of the first joint."""

from pathlib import Path
from shutil import copyfileobj
from urllib import request
from zipfile import ZipFile, ZipInfo

import numpy as np
from scipy.io import loadmat

from src.data.base import Table, TabularDataset


class SarcosDataset(TabularDataset):
    filename = "sarcos.zip"
    url = "https://gaussianprocess.org/gpml/data/"
    target = "joint_1_torque"
    available_folds = (0,)
    columns = tuple(f"{kind}_{joint}" for kind in ("position", "velocity", "acceleration")
                    for joint in range(1, 8))
    partitions = ("sarcos_inv", "sarcos_inv_test")

    @classmethod
    def fetch(cls, url: str, destination: Path) -> None:
        # Bundle both official files atomically. Fixed ZIP metadata makes the hash reproducible.
        with ZipFile(destination, "w") as archive:
            for key in cls.partitions:
                name = f"{key}.mat"
                with request.urlopen(url + name, timeout=60) as source:
                    with archive.open(ZipInfo(name), "w") as output:
                        copyfileobj(source, output)

    @classmethod
    def read(cls, path: Path) -> Table:
        tables = []
        with ZipFile(path) as archive:
            for key in cls.partitions:
                with archive.open(f"{key}.mat") as source:
                    matrix = loadmat(source, variable_names=[key])[key]
                if matrix.ndim != 2 or matrix.shape[1] != 28:
                    raise ValueError(f"{key} requires 21 inputs and seven torques")
                tables.append(matrix)
        values = np.concatenate(tables)
        is_test = np.arange(len(values)) >= len(tables[0])
        return values[:, :21], values[:, 21], is_test

    @classmethod
    def split_indices(cls, is_test: np.ndarray, fold: int) -> dict[str, np.ndarray]:
        pool = np.random.default_rng(13601).permutation(np.flatnonzero(~is_test))
        val, train = np.split(pool, [round(0.2 * len(pool))])
        return {"train": train, "val": val, "test": np.flatnonzero(is_test)}
