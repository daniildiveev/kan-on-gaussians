import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import ModelSettings
from src.data import load_dataset
from src.models import make_model


FILENAME = "concrete_compressive_strength.csv"


def test_matches_archived_protocol(tmp_path, concrete_csv):
    archive_root = Path(__file__).resolve().parents[1] / "archive"
    path = tmp_path / FILENAME
    path.write_bytes(concrete_csv)
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    x, y = table[:, :-1].copy(), table[:, -1]
    _, groups = np.unique(x[:, :7], axis=0, return_inverse=True)
    x[:, -1] = np.log1p(x[:, -1])
    # The historical study imports its own src package in its original working directory.
    script = """
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))
from archive import experiments, src
sys.modules.update(experiments=experiments, src=src)

from experiments.concrete_order import _folds
print(json.dumps(_folds(json.load(sys.stdin))))
"""
    archived_splits = subprocess.check_output(
        [sys.executable, "-c", script], cwd=archive_root,
        input=json.dumps(groups.tolist()), text=True,
    )
    for fold, indices in enumerate(json.loads(archived_splits)):
        data = load_dataset("concrete", root=tmp_path, fold=fold)
        assert data.n_features == 8
        train_x = x[indices["train"]]
        for name, rows in indices.items():
            features, target = getattr(data, name).tensors
            expected = (x[rows] - train_x.mean(0)) / train_x.std(0)
            np.testing.assert_allclose(features, expected, atol=1e-6)
            np.testing.assert_allclose(data.inverse_target(target), y[rows], atol=1e-4)

    batch, targets = next(iter(DataLoader(data.train, batch_size=64)))
    for name in ("mlp", "mgs_kan"):
        with torch.no_grad():
            assert make_model(name, ModelSettings(features=8))(batch).shape == targets.shape


@pytest.mark.parametrize("defect", ["header", "row_count", "age", "target"])
def test_invalid_concrete(tmp_path, concrete_csv, defect):
    lines = concrete_csv.decode().splitlines()
    if defect == "header":
        lines[0] = lines[0].replace("Cement", "Other")
    elif defect == "row_count":
        lines.pop()
    else:
        row = lines[1].split(",")
        row[7 if defect == "age" else 8] = "0"
        lines[1] = ",".join(row)
    (tmp_path / FILENAME).write_text("\n".join(lines))
    with pytest.raises(ValueError, match="Concrete"):
        load_dataset("concrete", root=tmp_path)


def test_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="unknown-name"):
        load_dataset("unknown-name", root=tmp_path)
