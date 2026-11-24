from pathlib import Path

from src.data.airfoil import AirfoilDataset
from src.data.appliances import AppliancesDataset
from src.data.base import TabularDataset
from src.data.concrete import ConcreteDataset
from src.data.csv_dataset import CsvDataset, CsvSpec
from src.data.gas_turbine import GasTurbineCODataset, GasTurbineNOxDataset
from src.data.power_plant import PowerPlantDataset
from src.data.sarcos import SarcosDataset
from src.data.superconductivity import SuperconductivityDataset
from src.data.wine_quality import WineQualityDataset
from src.paths import DATA_DIR


_DATASETS: dict[str, type[TabularDataset]] = {
    "airfoil": AirfoilDataset,
    "power_plant": PowerPlantDataset,
    "sarcos": SarcosDataset,
    "gas_turbine_co": GasTurbineCODataset,
    "gas_turbine_nox": GasTurbineNOxDataset,
    "superconductivity": SuperconductivityDataset,
    "appliances": AppliancesDataset,
    "concrete": ConcreteDataset,
    "wine_quality": WineQualityDataset,
}


def dataset_class(name: str) -> type[TabularDataset]:
    """Look up the reader and supported folds without opening or downloading data."""
    try:
        return _DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}. Available: {', '.join(_DATASETS)}") from None


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
    feature_columns: tuple[str, ...] | None = None,
) -> TabularDataset:
    """Load a registered dataset by name, or a one-off local CSV by specification."""
    if isinstance(dataset, CsvSpec):
        if feature_columns is not None:
            raise ValueError("Named feature selection requires a registered dataset")
        return CsvDataset._load_spec(dataset, root=root, fold=fold, download=download)

    data = dataset_class(dataset).load(root=root, fold=fold, download=download)
    return data if feature_columns is None else data.select_features(feature_columns)
