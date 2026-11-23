from abc import ABC, abstractmethod
from collections.abc import Callable
from inspect import isabstract
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar, Self
from urllib import request

import numpy as np

from .splits import DatasetSplits


Table = tuple[np.ndarray, np.ndarray, np.ndarray]


class TabularDataset(DatasetSplits, ABC):
    """A loaded regression dataset. Subclasses describe the source and implement read()."""

    filename: ClassVar[str]
    url: ClassVar[str | None] = None

    @classmethod
    @abstractmethod
    def read(cls, path: Path) -> Table:
        """Return raw features [N, D], targets [N] and group labels [N]."""

    @classmethod
    def load(
        cls,
        *,
        root: str | Path = "data",
        fold: int = 0,
        download: bool = False,
    ) -> Self:
        """Load from the cache, optionally fetching a missing source."""
        if isabstract(cls):
            raise TypeError(f"{cls.__name__} must implement read(path)")
        return cls._load_source(cls.filename, cls.read, root=root, fold=fold, download=download)

    @classmethod
    def _load_source(
        cls,
        filename: str,
        read: Callable[[Path], Table],
        *,
        root: str | Path,
        fold: int,
        download: bool,
    ) -> Self:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Dataset filename must be relative to root")

        path = Path(root) / relative
        if path.exists():
            return cls._load_file(path, read, fold)

        if cls.url is None:
            raise FileNotFoundError(f"Place the local dataset at {path}")
        if not download:
            raise FileNotFoundError(f"Expected {path}; use download=True to fetch the dataset")

        return cls._download_and_cache(path, cls.url, read, fold)

    @classmethod
    def _load_file(cls, path: Path, read: Callable[[Path], Table], fold: int) -> Self:
        features, targets, groups = read(path)
        return cls._from_arrays(features, targets, groups, fold)

    @classmethod
    def _download_and_cache(
        cls,
        path: Path,
        url: str,
        read: Callable[[Path], Table],
        fold: int,
    ) -> Self:
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as staging:
            downloaded_path = Path(staging) / path.name
            with request.urlopen(url, timeout=30) as response:
                downloaded_path.write_bytes(response.read())

            dataset = cls._load_file(downloaded_path, read, fold)
            try:
                path.hardlink_to(downloaded_path)
            except FileExistsError:
                # Another caller published its file first. Use that cached version.
                return cls._load_file(path, read, fold)
            return dataset
