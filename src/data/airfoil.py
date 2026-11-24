"""NASA airfoil noise, grouped by the four non-frequency test conditions."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np

from src.data.base import Table, TabularDataset


class AirfoilDataset(TabularDataset):
    filename = "airfoil.zip"
    url = "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip"
    columns = ("frequency", "attack_angle", "chord_length", "free_stream_velocity", "displacement_thickness")
    target = "scaled-sound-pressure"

    @classmethod
    def read(cls, path: Path) -> Table:
        with ZipFile(path) as archive, archive.open("airfoil_self_noise.dat") as source:
            values = np.loadtxt(source, ndmin=2)
        if values.shape[1] != 6:
            raise ValueError("Airfoil requires five inputs and sound pressure in dB")
        x = values[:, :5]
        return x, values[:, 5], cls.group_ids(x[:, 1:])
