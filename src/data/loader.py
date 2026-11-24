from pathlib import Path

from src.data.base import TabularDataset
from src.data.concrete import ConcreteDataset
from src.data.csv_dataset import CsvDataset, CsvSpec
from src.paths import DATA_DIR


_DATASETS: dict[str, type[TabularDataset]] = {"concrete": ConcreteDataset}


def register_dataset(name: str, dataset_class: type[TabularDataset]) -> None:
    """Expose a dataset class by name for this Python process."""
    if not isinstance(dataset_class, type) or not issubclass(dataset_class, TabularDataset):
        raise TypeError("Expected a TabularDataset subclass")
    if name in _DATASETS:
        raise ValueError(f"Dataset {name!r} is already registered")
    _DATASETS[name] = dataset_class


def load_dataset(
    dataset: str | CsvSpec,
    *,
    root: str | Path = DATA_DIR,
    fold: int = 0,
    download: bool = False,
) -> TabularDataset:
    """Load a registered dataset by name, or a one-off local CSV by specification."""
    if isinstance(dataset, CsvSpec):
        return CsvDataset._load_spec(dataset, root=root, fold=fold, download=download)

    if dataset not in _DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {', '.join(_DATASETS)}")
    return _DATASETS[dataset].load(root=root, fold=fold, download=download)
