from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class DatasetSplits:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    n_features: int
    _target_mean: float
    _target_scale: float
    fold: int | None = None
    source_file: str | None = None
    source_sha256: str | None = None
    target_column: str | None = None

    def inverse_target(self, values: torch.Tensor) -> torch.Tensor:
        """Convert standardized predictions or targets to original units."""
        return values * self._target_scale + self._target_mean

    @classmethod
    def _from_arrays(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        fold: int,
        *,
        source_file: str | None = None,
        source_sha256: str | None = None,
    ) -> Self:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        groups = np.asarray(groups)

        if x.ndim != 2 or x.shape[1] == 0 or y.shape != (len(x),) or groups.shape != y.shape:
            raise ValueError(
                "read() must return features [rows, columns], targets [rows] and groups [rows]"
            )
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Features and targets must be finite")
        if np.any(groups != groups):
            raise ValueError("Group labels must not be missing")

        rows = split_rows(groups, fold)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                features, targets, target_mean, target_scale = _standardize(x, y, rows["train"])
        except FloatingPointError as error:
            raise ValueError("Values exceed the numeric range for standardization") from error

        splits = {
            name: TensorDataset(features[indices], targets[indices])
            for name, indices in rows.items()
        }
        return cls(
            **splits,
            n_features=x.shape[1],
            _target_mean=target_mean,
            _target_scale=target_scale,
            fold=fold,
            source_file=source_file,
            source_sha256=source_sha256,
            target_column=getattr(cls, "target", None),
        )


def _standardize(
    x: np.ndarray,
    y: np.ndarray,
    train_rows: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Fit statistics on training rows and apply them to all rows."""
    train_x = x[train_rows]
    train_y = y[train_rows]

    feature_means = train_x[0].copy()
    feature_scales = np.ones(x.shape[1])
    varying = np.any(train_x != train_x[0], axis=0)
    if varying.any():
        varying_columns = train_x[:, varying]
        feature_means[varying] = varying_columns.mean(axis=0)
        feature_scales[varying] = varying_columns.std(axis=0)

    target_mean = float(train_y[0])
    target_scale = 1.0
    if np.any(train_y != train_y[0]):
        target_mean = float(train_y.mean())
        target_scale = float(train_y.std())

    scaled_x = (x - feature_means) / feature_scales
    scaled_y = (y - target_mean) / target_scale
    features = torch.from_numpy(scaled_x.astype(np.float32))
    targets = torch.from_numpy(scaled_y.astype(np.float32))
    return features, targets, target_mean, target_scale


def split_rows(groups: np.ndarray, fold: int) -> dict[str, np.ndarray]:
    """Assign whole groups using the frozen Concrete study's five-fold protocol."""
    if type(fold) is not int or not 0 <= fold < 5:
        raise ValueError("fold must be an integer from 0 through 4")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise ValueError("Five folds require at least five distinct groups")

    seed = 13601
    shuffled_groups = np.random.default_rng(seed).permutation(unique_groups)
    folds = np.array_split(shuffled_groups, 5)
    test_groups = folds[fold]

    remaining_groups = np.concatenate(folds[:fold] + folds[fold + 1 :])
    validation_rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 1]))
    remaining_groups = validation_rng.permutation(remaining_groups)
    validation_count = round(0.2 * len(remaining_groups))
    validation_groups = remaining_groups[:validation_count]
    train_groups = remaining_groups[validation_count:]

    return {
        "train": np.flatnonzero(np.isin(groups, train_groups)),
        "val": np.flatnonzero(np.isin(groups, validation_groups)),
        "test": np.flatnonzero(np.isin(groups, test_groups)),
    }
