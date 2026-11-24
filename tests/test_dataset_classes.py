import numpy as np
import pytest
import torch

from src.data import (
    ConcreteDataset, CsvDataset, CsvSpec, DatasetSplits, TabularDataset, load_dataset,
)


def test_csv_subclass_matches_one_off_spec(tmp_path):
    class HousingDataset(CsvDataset):
        filename = "houses.csv"
        target = "price"

    rows = np.arange(60).reshape(20, 3)
    np.savetxt(tmp_path / HousingDataset.filename, rows, delimiter=",", header="a,price,b", comments="")
    direct = HousingDataset.load(root=tmp_path, fold=2)
    generic = load_dataset(CsvSpec("houses.csv", "price"), root=tmp_path, fold=2)
    assert isinstance(direct, HousingDataset)
    assert isinstance(generic, CsvDataset)
    for split in ("train", "val", "test"):
        for left, right in zip(getattr(direct, split).tensors, getattr(generic, split).tensors):
            assert torch.equal(left, right)


def test_custom_format_uses_shared_pipeline(tmp_path):
    class ArrayDataset(TabularDataset):
        filename = "rows.npy"

        @classmethod
        def read(cls, path):
            rows = np.load(path)
            return rows[:, :1], rows[:, 1], np.arange(len(rows))

    rows = np.arange(40).reshape(20, 2)
    np.save(tmp_path / "rows.npy", rows)
    data = ArrayDataset.load(root=tmp_path)
    assert isinstance(data, ArrayDataset)
    assert data.n_features == 1
    targets = torch.cat([data.inverse_target(s.tensors[1]) for s in (data.train, data.val, data.test)])
    np.testing.assert_allclose(targets.sort().values, rows[:, 1], atol=1e-5)
    x, _ = data.train.tensors
    torch.testing.assert_close(x.std(dim=0, correction=0), torch.ones(1))


def test_concrete_factory_returns_loaded_subclass(tmp_path, concrete_csv):
    (tmp_path / ConcreteDataset.filename).write_bytes(concrete_csv)
    direct = ConcreteDataset.load(root=tmp_path)
    named = load_dataset("concrete", root=tmp_path)
    assert isinstance(named, ConcreteDataset)
    assert isinstance(named, DatasetSplits)
    assert torch.equal(direct.train.tensors[0], named.train.tensors[0])


def test_incomplete_subclass_cannot_load(tmp_path):
    class MissingReader(TabularDataset):
        filename = "empty"

    with pytest.raises(TypeError):
        MissingReader.load(root=tmp_path)


def test_registered_name_resolves_custom_dataset(tmp_path, monkeypatch):
    from src.data import loader, register_dataset

    class LocalDataset(CsvDataset):
        filename = "table.csv"
        target = "label"

    monkeypatch.setattr(loader, "_DATASETS", {"concrete": ConcreteDataset})
    register_dataset("local", LocalDataset)
    rows = np.arange(30).reshape(15, 2)
    np.savetxt(tmp_path / "table.csv", rows, delimiter=",", header="feature,label", comments="")
    assert isinstance(load_dataset("local", root=tmp_path), LocalDataset)
    with pytest.raises(ValueError, match="local"):
        register_dataset("local", ConcreteDataset)
