import numpy as np
import pytest
import torch

from mgs_kan.data import CsvSpec, load_dataset


HOUSES = CsvSpec(filename="houses.csv", target="price")


def write_houses(root, rows):
    np.savetxt(root / HOUSES.filename, rows, delimiter=",", header="a,price,b", comments="")


def test_local_csv_folds_and_scaling(tmp_path):
    ids = np.arange(30)
    rows = np.column_stack((ids**2, ids, -ids))
    write_houses(tmp_path, rows)
    held_out = []
    for fold in range(5):
        data = load_dataset(HOUSES, root=tmp_path, fold=fold)
        splits = (data.train, data.val, data.test)
        indices = [data.inverse_target(s.tensors[1]).round().long().numpy() for s in splits]
        assert sorted(np.concatenate(indices)) == list(ids)
        held_out.extend(indices[-1])
        features = rows[:, [0, 2]]
        train_x = features[indices[0]]
        for split, selected in zip(splits, indices):
            x, y = split.tensors
            assert x.shape == (len(selected), data.n_features) and data.n_features == 2
            assert y.shape == (len(selected),)
            assert x.dtype == y.dtype == torch.float32
            expected = (features[selected] - train_x.mean(0)) / train_x.std(0)
            np.testing.assert_allclose(x, expected, atol=1e-6)
            np.testing.assert_allclose(y, (selected - indices[0].mean()) / indices[0].std(), atol=1e-6)
    assert sorted(held_out) == list(ids)

    rows[indices[-1]] += 1000
    write_houses(tmp_path, rows)
    repeated = load_dataset(HOUSES, root=tmp_path, fold=4)
    for old, new in zip(data.train.tensors, repeated.train.tensors):
        assert torch.equal(old, new)


def test_constant_feature_and_target(tmp_path):
    write_houses(tmp_path, np.full((20, 3), 7))
    data = load_dataset(HOUSES, root=tmp_path)
    x, y = data.train.tensors
    assert torch.count_nonzero(x) == torch.count_nonzero(y) == 0
    assert torch.all(data.inverse_target(y) == 7)


@pytest.mark.parametrize("csv", [
    "", "price\n1\n", ",price\n1,2\n", "a,b\n1,2\n", "a,a,price\n1,1,2\n",
    "a,price\n1,\n", "a,price\n1,cat\n", "a,price\ninf,2\n", "a,price\n1,2,3\n",
])
def test_rejects_invalid_table(tmp_path, csv):
    (tmp_path / HOUSES.filename).write_text(csv)
    with pytest.raises(ValueError):
        load_dataset(HOUSES, root=tmp_path)


@pytest.mark.parametrize("fold", [-1, 5, 0.5])
def test_invalid_fold(tmp_path, fold):
    write_houses(tmp_path, np.ones((10, 3)))
    with pytest.raises(ValueError, match="fold"):
        load_dataset(HOUSES, root=tmp_path, fold=fold)


def test_too_few_groups(tmp_path):
    write_houses(tmp_path, np.ones((3, 3)))
    with pytest.raises(ValueError, match="groups"):
        load_dataset(HOUSES, root=tmp_path)


def test_repeated_decimal_uses_unit_scale(tmp_path):
    rows = np.full((5, 3), 0.1)
    rows[:, 1] = np.arange(5)
    write_houses(tmp_path, rows)
    probe = load_dataset(HOUSES, root=tmp_path)
    held_out = probe.inverse_target(probe.test.tensors[1]).round().long().item()
    rows[:, 1] = 0.1
    rows[held_out] = 1.1
    write_houses(tmp_path, rows)
    data = load_dataset(HOUSES, root=tmp_path)
    x, y = data.train.tensors
    assert torch.all(x == 0) and torch.all(y == 0)
    test_x, test_y = data.test.tensors
    torch.testing.assert_close(test_x, torch.ones_like(test_x))
    torch.testing.assert_close(test_y, torch.ones_like(test_y))
    torch.testing.assert_close(data.inverse_target(test_y), torch.tensor([1.1]))
