from itertools import combinations

import pytest
import torch
from torch.utils.data import TensorDataset

from src.config import ModelSettings
from src.data import DatasetSplits
from src.models import make_model, parameter_count


def test_full_precision_triples_and_parameter_budget():
    settings = ModelSettings(features=9, single_kernels=9, pair_kernels=5, init_seed=33)
    model = make_model("hypergraph_mgs_kan", settings).double()
    assert parameter_count(model) == 2173
    assert model.triples.tolist() == [list(g) for g in combinations(range(9), 3)]
    x = torch.randn(4, 9, dtype=torch.double)
    coords = x[:, model.triples]
    factor = torch.zeros(84, 3, 3, dtype=x.dtype)
    row, col = torch.tril_indices(3, 3)
    factor[:, row, col] = torch.where(row == col, model.triple_factor_raw.exp(), model.triple_factor_raw)
    projected = torch.einsum("bni,nij->bnj", coords - model.triple_center, factor)
    expected = model.triple_amplitude * torch.exp(-0.5 * projected.square().sum(-1))
    torch.testing.assert_close(model.triple_values(coords), expected)
    affine, singles, pairs = model.contributions(x)
    torch.testing.assert_close(model(x), affine + singles.sum(-1) + pairs.sum(-1) + expected.sum(-1))
    model(x).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match="three features"):
        make_model("hypergraph_mgs_kan", ModelSettings(features=2))


def test_validation_ranking_matches_direct_term_removal_and_target_units():
    pytest.importorskip("mlflow")
    from src.experiments.triples import rank_validation_triples

    model = make_model("hypergraph_mgs_kan", ModelSettings(features=5, init_seed=11))
    x = torch.randn(100, 5, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        model.triple_amplitude.zero_()
        model.triple_amplitude[2] = 1
        y = model(x)
    partition = TensorDataset(x, y)
    data = DatasetSplits(train=partition, val=partition, test=partition, n_features=5,
                         _target_mean=32., _target_scale=7., selected_features=tuple("ABCDE"))
    ranking = rank_validation_triples(model, data, batch_size=23)
    assert ranking.iloc[0].triple == "A / B / E"
    assert ranking.iloc[0].mixed_strength > 0
    assert ranking.iloc[0]["rank"] == 1 and ranking.iloc[1:].mixed_strength.eq(0).all()
    with torch.no_grad():
        model.triple_amplitude[2] = 0
        expected = ((model(x) - y).square().mean().sqrt() * 7).item()
    assert ranking.iloc[0].delta_val_rmse == pytest.approx(expected, rel=1e-5)
