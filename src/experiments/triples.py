"""Rank individual triple terms by their ablation on the validation partition."""

from itertools import product

import pandas as pd
import torch

from src.data import DatasetSplits
from src.models.hypergraph import HypergraphMGS_KAN


@torch.inference_mode()
def rank_validation_triples(
    model: HypergraphMGS_KAN, data: DatasetSplits, batch_size: int = 64, reference_seed: int = 2013,
) -> pd.DataFrame:
    """Positive delta means removing this term increases RMSE, with all other weights fixed."""
    features, targets = data.val.tensors
    columns = data.selected_features or data.columns
    device = model.triple_center.device
    # Pair observed validation rows identically for every fit; mixed differences use eight corners.
    reference = torch.randperm(len(features), generator=torch.Generator().manual_seed(reference_seed))
    ablated_sse = torch.zeros(len(model.triples), dtype=torch.float64, device=device)
    mixed_squares = torch.zeros_like(ablated_sse)
    full_sse = torch.zeros((), dtype=torch.float64, device=device)
    model.eval()
    for start in range(0, len(features), batch_size):
        batch = slice(start, start + batch_size)
        x = features[batch].to(device)
        observed = x[:, model.triples]
        alternate = features[reference[batch]].to(device)[:, model.triples]
        terms = model.triple_values(observed).double()
        pred = model(x).double()
        actual = data.inverse_target(targets[batch].to(device).double())
        full_sse += (data.inverse_target(pred) - actual).square().sum()
        removed = data.inverse_target(pred[:, None] - terms)
        ablated_sse += (removed - actual[:, None]).square().sum(0)

        mixed = torch.zeros_like(terms)
        for mask in product((False, True), repeat=3):
            corners = torch.where(torch.tensor(mask, device=device), alternate, observed)
            mixed += (-1) ** sum(mask) * model.triple_values(corners).double()
        mixed = data.inverse_target(mixed) - data.inverse_target(torch.zeros_like(full_sse))
        mixed_squares += mixed.square().sum(0)

    full_rmse = (full_sse / len(features)).sqrt()
    delta = (ablated_sse / len(features)).sqrt() - full_rmse
    strength = (mixed_squares / (8 * len(features))).sqrt()
    if not torch.isfinite(delta).all() or not torch.isfinite(strength).all():
        raise ValueError("Nonfinite validation triple scores")
    scores = pd.DataFrame({
        "triple": [" / ".join(columns[i] for i in group) for group in model.triples.tolist()],
        "delta_val_rmse": delta.cpu().numpy(),
        "mixed_strength": strength.cpu().numpy(),
    }).sort_values(["delta_val_rmse", "triple"], ascending=[False, True], ignore_index=True)
    scores["rank"] = scores.index + 1
    return scores
