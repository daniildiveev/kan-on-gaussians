import pandas as pd
import pytest
import torch

pytest.importorskip("mlflow")

from src.config import ModelSettings
from src.experiments.search import AIRFOIL_STAGES, ValidationGrid
from src.models import make_model, parameter_count


def test_profile_sizes_and_airfoil_budgets():
    assert ValidationGrid.preset("airfoil_first").size == 12
    assert ValidationGrid.preset("airfoil_all").size == 18
    deep = ValidationGrid.preset("deep")
    assert deep.size == 54 and deep.folds == (0,)
    for stage, expected in zip(AIRFOIL_STAGES, [(106, 111), (181, 186), (346, 351)], strict=True):
        settings = stage.settings.model_copy(update={"features": 5})
        assert tuple(parameter_count(make_model(m, settings)) for m in stage.models) == expected


@pytest.mark.parametrize("name", ["deep_mlp", "resnet"])
def test_deep_models_init_and_batch_of_one(name):
    settings = ModelSettings(features=5, init_seed=17, deep_width=128, deep_depth=4)
    model = make_model(name, settings)
    replica = make_model(name, settings)
    torch.testing.assert_close(model.state_dict(), replica.state_dict())
    output = model(torch.ones(1, 5))
    assert output.shape == (1,)
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_selection_requires_paired_improvements():
    grid = ValidationGrid.preset("airfoil_first")
    rows = []
    for model, second_scores in [("mlp", [0.7, 1.1, 1.1]), ("mgs_kan", [0.98, 0.98, 1.0])]:
        for stage, size, scores in [("stage-1", 100, [1, 1, 1]), ("stage-2", 180, second_scores)]:
            for (init, shuffle), rmse in zip(grid.seeds, scores, strict=True):
                rows.append(dict(dataset="airfoil", model=model, fold=0, stage=stage, parameters=size,
                                 init_seed=init, shuffle_seed=shuffle, val_rmse=rmse, seconds=1, best_epoch=1))
    runs = pd.DataFrame(rows)
    comparison = grid.compare(runs)
    assert dict(comparison[comparison.selected].set_index("model").stage) == {
        "mlp": "stage-1", "mgs_kan": "stage-2",
    }
    with pytest.raises(ValueError, match="complete"):
        grid.compare(runs.iloc[:-1])
