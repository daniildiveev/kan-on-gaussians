"""Tabular regression datasets for PyTorch."""

from src.data.airfoil import AirfoilDataset
from src.data.appliances import AppliancesDataset
from src.data.base import TabularDataset
from src.data.concrete import ConcreteDataset
from src.data.csv_dataset import CsvDataset, CsvSpec
from src.data.gas_turbine import GasTurbineCODataset, GasTurbineNOxDataset
from src.data.loader import dataset_class, load_dataset, register_dataset
from src.data.power_plant import PowerPlantDataset
from src.data.sarcos import SarcosDataset
from src.data.splits import DatasetSplits
from src.data.superconductivity import SuperconductivityDataset
from src.data.wine_quality import WineQualityDataset

__all__ = [
    "AirfoilDataset", "AppliancesDataset", "GasTurbineCODataset", "GasTurbineNOxDataset",
    "PowerPlantDataset", "SarcosDataset", "SuperconductivityDataset", "WineQualityDataset",
    "ConcreteDataset", "CsvDataset", "CsvSpec", "DatasetSplits", "TabularDataset",
    "dataset_class", "load_dataset", "register_dataset",
]
