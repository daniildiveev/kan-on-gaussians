"""Pre-training checks for the model, generators, and interaction metric."""

from itertools import combinations

import numpy as np
import torch
from torch.func import functional_call

from . import config as c
from .data import gaussian, make_data, target
from .models import MGS_KAN, make_model, parameter_count


def run_checks():
    torch.manual_seed(31415)
    model = MGS_KAN().double()
    raw = model.pair_precision_raw
    with torch.no_grad():
        raw.add_(0.3 * torch.randn_like(raw))
    assert model.pairs.tolist() == [list(p) for p in combinations(range(c.FEATURES), 2)]
    assert model.precision().shape == (66, c.PAIR_KERNELS, 2, 2)
    assert torch.linalg.eigvalsh(model.precision()).min() > 0
    counts = {name: parameter_count(make_model(name)) for name in c.MODELS}
    assert counts == {"mgs_kan": 949, "mlp": 937}
    assert abs(counts["mgs_kan"] - counts["mlp"]) / counts["mgs_kan"] <= 0.05

    x = torch.randn(5, c.FEATURES, dtype=torch.float64, requires_grad=True)
    names, parameters = zip(*model.named_parameters())
    assert torch.autograd.gradcheck(
        lambda x, *p: functional_call(model, dict(zip(names, p)), (x,)),
        (x, *parameters), fast_mode=True,
    )
    assert torch.autograd.gradcheck(
        lambda a: torch.linalg.vector_norm(a, dim=1).sum(),
        (model.pair_amplitude.detach().requires_grad_(True),),
    )

    pair = c.PLANTED[-1]
    index = model.pairs.tolist().index(list(pair))
    center = torch.tensor(c.GAUSSIAN_CENTER, dtype=torch.float64)
    precision = torch.tensor(c.GAUSSIAN_PRECISION, dtype=torch.float64)
    factor = torch.linalg.cholesky(precision)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.pair_center[index, 0].copy_(center)
        model.pair_amplitude[index, 0] = c.GAUSSIAN_AMPLITUDE
        raw[index, 0].copy_(torch.stack((factor[0, 0].log(), factor[1, 0], factor[1, 1].log())))
        grid = torch.linspace(-2.5, 2.5, 23, dtype=torch.float64)
        probes = torch.zeros(23**2, c.FEATURES, dtype=torch.float64)
        probes[:, pair] = torch.cartesian_prod(grid, grid)
        actual = model(probes)
        expected = torch.from_numpy(gaussian(probes[:, pair].numpy()))
        gaussian_error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        # Reflection flips the precision cross-term. This catches an axis swap or sign error.
        reflected = probes.clone()
        reflected[:, pair[1]] = 2 * center[1] - reflected[:, pair[1]]
        assert (actual - model(reflected)).abs().max() > 0.1

        # Compare the analytic marginal subtraction against independent quadrature.
        mean = torch.linspace(-0.35, 0.25, c.FEATURES, dtype=torch.float64)
        sd = torch.linspace(0.65, 1.35, c.FEATURES, dtype=torch.float64)
        delta = probes[:40, pair] - center
        nodes, weights = np.polynomial.hermite.hermgauss(96)
        nodes = torch.tensor(nodes * np.sqrt(2))
        weights = torch.tensor(weights / np.sqrt(np.pi))
        a = (mean[pair[0]] + sd[pair[0]] * nodes - center[0])
        b = (mean[pair[1]] + sd[pair[1]] * nodes - center[1])
        p, q, r = precision[0, 0], precision[0, 1], precision[1, 1]

        def value(u, v):
            return c.GAUSSIAN_AMPLITUDE * torch.exp(-0.5 * (p*u.square() + 2*q*u*v + r*v.square()))

        pure = (value(*delta.T) - value(delta[:, :1], b) @ weights
                - value(a, delta[:, 1:]) @ weights + weights @ value(a[:, None], b) @ weights)
        actual_pure = model.pure_interactions(probes[:40], mean, sd)
        torch.testing.assert_close(actual_pure[:, index], pure, atol=2e-10, rtol=2e-10)
        assert torch.count_nonzero(actual_pure[:, [i for i in range(66) if i != index]]) == 0
        # An effectively single-feature kernel should contribute essentially zero.
        raw[index, 0, 1] = 0
        raw[index, 0, 2] = -16
        assert model.pure_interactions(probes[:40], mean, sd).abs().max() < 1e-12

    sparse = make_data("sparse", 314, (256, 64, 128))
    additive = make_data("additive", 314, (256, 64, 128))
    repeat = make_data("sparse", 314, (256, 64, 128))
    for (x1, y1, f1), (x2, y2, f2), (xr, yr, fr) in zip(sparse[0], additive[0], repeat[0]):
        assert torch.equal(x1, x2) and torch.equal(x1, xr)
        assert torch.equal(y1, yr) and torch.equal(f1, fr)
        torch.testing.assert_close(y1 - f1, y2 - f2, atol=6e-7, rtol=0)
    train_x = sparse[0][0][0]
    torch.testing.assert_close(train_x.mean(dim=0), torch.zeros(c.FEATURES), atol=1e-7, rtol=0)
    torch.testing.assert_close(train_x.std(dim=0, correction=0), torch.ones(c.FEATURES))
    random_features = np.random.default_rng(101).standard_normal((50_000, c.FEATURES))
    for name in c.DATASETS:
        response = target(name, random_features)
        assert abs(response.mean()) < 0.018 and abs(response.std() - 1) < 0.018
    return {"status": "passed", "dtype": "float64 (model checks)", "parameter_counts": counts,
            "candidate_pairs": len(model.pairs), "rotated_gaussian_max_abs_error": gaussian_error,
            "fixture": "substitute with nonzero precision cross-term"}


if __name__ == "__main__":
    print(run_checks())
