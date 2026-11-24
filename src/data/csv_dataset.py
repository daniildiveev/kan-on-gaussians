import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Self

import numpy as np

from src.data.base import Table, TabularDataset


@dataclass(frozen=True)
class CsvSpec:
    """A numeric CSV under root, with one named regression target."""

    filename: str
    target: str


class CsvDataset(TabularDataset):
    """Set filename and target for a numeric CSV, or override prepare_columns()."""

    target: ClassVar[str]

    @classmethod
    def read(cls, path: Path) -> Table:
        header, values = cls._read_csv(path)
        return cls.prepare_columns(header, values)

    @classmethod
    def prepare_columns(cls, header: list[str], values: np.ndarray) -> Table:
        """Use non-target columns in header order and give each row its own group."""
        return cls._select_target(header, values, cls.target)

    @classmethod
    def _load_spec(
        cls,
        spec: CsvSpec,
        *,
        root: str | Path,
        fold: int,
        download: bool,
    ) -> Self:
        def read(path: Path) -> Table:
            header, values = cls._read_csv(path)
            return cls._select_target(header, values, spec.target)

        data = cls._load_source(spec.filename, read, root=root, fold=fold, download=download)
        return replace(data, target_column=spec.target)

    @staticmethod
    def _select_target(header: list[str], values: np.ndarray, target: str) -> Table:
        if target not in header:
            raise ValueError(f"Missing target column {target!r}")
        target_index = header.index(target)
        features = np.delete(values, target_index, axis=1)
        targets = values[:, target_index]
        groups = np.arange(len(values))
        return features, targets, groups

    @classmethod
    def _read_csv(cls, path: Path) -> tuple[list[str], np.ndarray]:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            try:
                header = [name.strip() for name in next(reader, [])]
                column_count = len(header)
                has_duplicate_names = len(set(header)) != column_count
                if column_count < 2 or not all(header) or has_duplicate_names:
                    raise ValueError(f"{path}: expected at least two distinct, nonempty column names")

                rows = []
                for row in reader:
                    numbers = cls._parse_row(row, column_count, path, reader.line_num)
                    rows.append(numbers)
            except csv.Error as error:
                raise ValueError(f"{path}:{reader.line_num}: {error}") from error

        values = np.asarray(rows, dtype=np.float64)
        if not rows or not np.isfinite(values).all():
            raise ValueError(f"{path}: expected nonempty, finite numeric data")
        return header, values

    @staticmethod
    def _parse_row(
        row: list[str],
        column_count: int,
        path: Path,
        line_number: int,
    ) -> list[float]:
        if len(row) != column_count:
            raise ValueError(f"{path}:{line_number}: expected {column_count} columns")
        try:
            return [float(value) for value in row]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: expected numeric values") from error
