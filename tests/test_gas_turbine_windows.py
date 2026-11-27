from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import ModelSettings
from src.data import dataset_class, load_dataset
from src.models import make_model, parameter_count


@pytest.mark.parametrize("dataset", ["gas_turbine_co", "gas_turbine_nox"])
def test_windows_preserve_year_order_targets_and_point_inputs(tmp_path, dataset):
    reader = dataset_class(dataset)
    with ZipFile(tmp_path / reader.filename, "w") as archive:
        for year, size in zip(range(2011, 2016), [3, 8, 6, 9, 7]):
            values = np.arange(size * 9).reshape(size, 9) + 100 * (year - 2011)
            frame = pd.DataFrame(values, columns=reader.columns)
            frame = frame.assign(CO=values[:, 0], NOX=-values[:, 1])
            archive.writestr(f"gt_{year}.csv", frame.to_csv(index=False))

    raw = load_dataset(dataset, root=tmp_path)
    data = load_dataset(dataset + "_window6", root=tmp_path)
    test_ends = [5, 6, 7, 8, 14, 15]
    for split, ends in [("train", [8, 9, 10]), ("val", [5]), ("test", test_ends)]:
        x, y = getattr(raw, split).tensors
        windows, targets = getattr(data, split).tensors
        torch.testing.assert_close(windows, torch.stack([x[end - 5:end + 1] for end in ends]))
        torch.testing.assert_close(data.inverse_target(targets), raw.inverse_target(y[ends]))
    x, y = data.test_endpoints().tensors
    torch.testing.assert_close(x, raw.test.tensors[0][test_ends])
    torch.testing.assert_close(y, raw.test.tensors[1][test_ends])
    subset = data.select_features(("TEY", "AT"))
    torch.testing.assert_close(subset.test.tensors[0], data.test.tensors[0][:, :, [7, 0]])


@pytest.mark.parametrize("name, count", [("gru_window6", 2201), ("mlp_window6", 2159)])
def test_sequence_model_regresses_one_scalar_per_window(name, count):
    settings = ModelSettings(features=9, init_seed=11, deep_width=22, mlp_width=26)
    model = make_model(name, settings)
    assert parameter_count(model) == count
    inputs = torch.randn(3, 6, 9)
    predictions = model(inputs)
    assert predictions.shape == (3,)
    torch.testing.assert_close(predictions, make_model(name, settings)(inputs))
    torch.testing.assert_close(predictions[:1], model(inputs[:1]))
    predictions.sum().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    with pytest.raises(ValueError, match="window"):
        model(inputs[:, -1])
