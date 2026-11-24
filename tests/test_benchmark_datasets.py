from io import BytesIO
from urllib import request
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from src.data import dataset_class, load_dataset


def test_airfoil_keeps_frequency_sweeps_together(tmp_path):
    reader = dataset_class("airfoil")
    conditions = np.repeat(np.arange(10), 2)
    frequency = np.tile([200, 800], 10)
    values = np.column_stack((frequency, conditions, conditions, conditions, conditions, conditions + 90))
    raw = BytesIO()
    np.savetxt(raw, values, delimiter="\t")
    path = tmp_path / reader.filename
    with ZipFile(path, "w") as archive:
        archive.writestr("airfoil_self_noise.dat", raw.getvalue())

    x, y, groups = reader.read(path)
    np.testing.assert_array_equal(x[:, 0], frequency)
    np.testing.assert_array_equal(y, conditions + 90)
    for fold in reader.available_folds:
        splits = reader.split_indices(groups, fold)
        group_sets = [set(conditions[rows]) for rows in splits.values()]
        assert sum(map(len, group_sets)) == len(set(conditions))
        data = load_dataset("airfoil", root=tmp_path, fold=fold)
        assert data.n_features == 5


def test_power_plant_uses_single_table_and_groups_duplicates(tmp_path):
    reader = dataset_class("power_plant")
    path = tmp_path / reader.filename
    path.write_text("AT,V,AP,RH,PE\n20,40,1000,60,440\n20,40,1000,60,445\n")
    x, y, groups = reader.read(path)
    assert x.shape == (2, 4)
    assert list(y) == [440, 445]
    assert groups[0] == groups[1]


def test_sarcos_preserves_official_test_and_cache(tmp_path, monkeypatch):
    reader = dataset_class("sarcos")
    file_data = {}
    for key, start, size in [("sarcos_inv", 0, 10), ("sarcos_inv_test", 100, 3)]:
        matrix = np.arange(size * 28).reshape(size, 28) + start
        buffer = BytesIO()
        savemat(buffer, {key: matrix})
        file_data[f"{key}.mat"] = buffer.getvalue()

    def fetch(url, timeout):
        return BytesIO(file_data[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(request, "urlopen", fetch)
    data = load_dataset("sarcos", root=tmp_path, download=True)
    assert [len(data.train), len(data.val), len(data.test)] == [8, 2, 3]
    assert data.n_features == 21
    np.testing.assert_allclose(data.inverse_target(data.test.tensors[1]), [121, 149, 177])
    assert reader.available_folds == (0,)
    with pytest.raises(ValueError, match="fold"):
        load_dataset("sarcos", root=tmp_path, fold=1)
    second = load_dataset("sarcos", root=tmp_path / "second-cache", download=True)
    assert second.source_sha256 == data.source_sha256


def test_gas_tasks_share_inputs_and_hold_out_last_two_years(tmp_path):
    co = dataset_class("gas_turbine_co")
    nox = dataset_class("gas_turbine_nox")
    path = tmp_path / co.filename
    with ZipFile(path, "w") as archive:
        for year in range(2011, 2016):
            row = dict.fromkeys(co.features, year)
            row.update(CO=year - 2010, NOX=year + 100)
            archive.writestr(f"gt_{year}.csv", pd.DataFrame([row]).to_csv(index=False))
    x, y, years = co.read(path)
    other_x, other_y, _ = nox.read(path)
    np.testing.assert_array_equal(x, other_x)
    np.testing.assert_array_equal(x, np.repeat(years[:, None], 9, axis=1))
    np.testing.assert_array_equal(y, years - 2010)
    np.testing.assert_array_equal(other_y, years + 100)
    splits = co.split_indices(years, 0)
    assert list(years[splits["train"]]) == [2011, 2012]
    assert list(years[splits["val"]]) == [2013]
    assert list(years[splits["test"]]) == [2014, 2015]


def test_superconductivity_groups_equivalent_formulas(tmp_path):
    reader = dataset_class("superconductivity")
    path = tmp_path / reader.filename
    data = pd.DataFrame(np.ones((3, 81)), columns=reader.columns)
    data["critical_temp"] = [10, 11, 20]
    formulas = pd.DataFrame(dict(H=[1, 2, 1], O=[1, 2, 2], material=["HO", "H2O2", "HO2"],
                                 critical_temp=data["critical_temp"]))
    with ZipFile(path, "w") as archive:
        archive.writestr("train.csv", data.to_csv(index=False))
        archive.writestr("unique_m.csv", formulas.to_csv(index=False))
    x, _, groups = reader.read(path)
    assert x.shape == (3, 81)
    assert groups[0] == groups[1] != groups[2]


def test_appliances_split_in_time_order_without_random_controls(tmp_path):
    reader = dataset_class("appliances")
    path = tmp_path / reader.filename
    dates = pd.date_range("2016-01-10", periods=20, freq="10min")
    data = pd.DataFrame(1, index=dates, columns=reader.room_features + reader.weather_features)
    data = data.assign(date=dates, Appliances=np.arange(20), rv1=999, rv2=999, lights=888)
    with ZipFile(path, "w") as archive:
        archive.writestr("energydata_complete.csv", data.iloc[::-1].to_csv(index=False))
    x, y, timestamps = reader.read(path)
    assert x.shape == (20, 26)
    assert np.max(x) == 1
    rows = reader.split_indices(timestamps, 0)
    assert list(y[rows["train"]]) == list(range(14))
    assert list(y[rows["val"]]) == [14, 15, 16]
    assert list(y[rows["test"]]) == [17, 18, 19]


def test_wine_groups_identical_measurements_across_colors(tmp_path):
    reader = dataset_class("wine_quality")
    path = tmp_path / reader.filename
    data = pd.DataFrame([dict.fromkeys(reader.features, 1.5)])
    with ZipFile(path, "w") as archive:
        for color, score in [("red", 6), ("white", 7)]:
            archive.writestr(f"winequality-{color}.csv",
                             data.assign(quality=score).to_csv(index=False, sep=";"))
    x, y, groups = reader.read(path)
    assert x.shape == (2, 11)
    assert list(y) == [6, 7]
    assert groups[0] == groups[1]
