from http.client import IncompleteRead
from io import BytesIO
from urllib import request

import pytest
import torch

from mgs_kan.data import CsvSpec, load_dataset


FILENAME = "concrete_compressive_strength.csv"


def forbid_network(*args, **kwargs):
    pytest.fail("Unexpected network access")


def test_missing_files_stay_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(request, "urlopen", forbid_network)
    with pytest.raises(FileNotFoundError, match="download=True") as info:
        load_dataset("concrete", root=tmp_path)
    assert str(tmp_path / FILENAME) in str(info.value)
    with pytest.raises(FileNotFoundError, match="houses.csv"):
        load_dataset(CsvSpec("houses.csv", "price"), root=tmp_path, download=True)


def test_first_download_and_cached_read(tmp_path, concrete_csv, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setattr(request, "urlopen", lambda *args, **kwargs: BytesIO(concrete_csv))
    first = load_dataset("concrete", root=root, download=True)
    assert (root / FILENAME).read_bytes() == concrete_csv
    monkeypatch.setattr(request, "urlopen", forbid_network)
    second = load_dataset("concrete", root=root, download=True)
    assert torch.equal(first.train.tensors[0], second.train.tensors[0])


@pytest.mark.parametrize("failure", ["interrupted", "invalid"])
def test_failed_download_can_retry(tmp_path, concrete_csv, monkeypatch, failure):
    class Interrupted(BytesIO):
        def read(self, *args):
            raise IncompleteRead(b"partial response")

    response = Interrupted() if failure == "interrupted" else BytesIO(b"x,y\n1,2\n")
    monkeypatch.setattr(request, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises((IncompleteRead, ValueError)):
        load_dataset("concrete", root=tmp_path, download=True)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(request, "urlopen", lambda *args, **kwargs: BytesIO(concrete_csv))
    assert load_dataset("concrete", root=tmp_path, download=True).n_features == 8


def test_existing_invalid_file_is_preserved(tmp_path, monkeypatch):
    path = tmp_path / FILENAME
    path.write_bytes(b"damaged")
    monkeypatch.setattr(request, "urlopen", forbid_network)
    with pytest.raises(ValueError):
        load_dataset("concrete", root=tmp_path, download=True)
    assert path.read_bytes() == b"damaged"
