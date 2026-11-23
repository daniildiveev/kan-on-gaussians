# MGS-KAN

This repository studies Multidimensional Gaussian Surface Kolmogorov–Arnold Networks.
The active package exposes two baselines from the scalar study. By default, both
accept batches of 12 features and return one prediction per example.

| Factory name | Model | Trainable parameters |
| --- | --- | ---: |
| `mlp` | Two ReLU hidden layers of width 24 | 937 |
| `mgs_kan` | Affine term plus single-feature and pairwise Gaussian kernels | 949 |

The active package provides scalar regression models and tabular datasets for
comparisons on real data. The published study
protocols, training code and recorded results live in the archive.

## Install and check

Use Python 3.11+ in a project environment. Install the test extra to run pytest.
For historical notebook environments, see the archive guide.

```sh
python -m pip install -e '.[test]'
python -m pytest
```

```python
import torch
from mgs_kan.config import ModelSettings
from mgs_kan.models import make_model

model = make_model("mgs_kan", ModelSettings(features=8))
with torch.no_grad():
    predictions = model(torch.zeros(2, 8))
```

`make_model("mgs_kan")` reads settings from the environment at construction.
For example, `MGS_KAN_FEATURES=8` selects eight inputs for either model.
Explicit field values override environment variables. Defaults reproduce the study models.

## Load regression data

```python
import torch
from torch.utils.data import DataLoader
from mgs_kan.config import ModelSettings
from mgs_kan.data import CsvSpec, load_dataset
from mgs_kan.models import make_model

data = load_dataset("concrete", root="data", fold=0, download=True)
model = make_model("mlp", ModelSettings(features=data.n_features))
features, targets = next(iter(DataLoader(data.train, batch_size=64, shuffle=True)))
with torch.no_grad():
    predictions_mpa = data.inverse_target(model(features))
    targets_mpa = data.inverse_target(targets)
```

The first download saves `data/concrete_compressive_strength.csv`. Later calls
reuse it. Downloads default to off. Concrete has five folds (0–4) grouped by
mixture recipe. Age is transformed with `log1p`. Credit: I‑Cheng Yeh,
*Concrete Compressive Strength*, UCI Machine Learning Repository, CC BY 4.0.
[Dataset and attribution](https://doi.org/10.24432/C5PK67)

For a local table, use `load_dataset(CsvSpec(filename="houses.csv", target="price"), root="data")`.
Supply a header and finite numeric values. All other columns are features in
header order. Each row is its own group, so this split suits independent observations.
Both sources provide `train`, `val` and `test` as PyTorch datasets. Standardization
uses the training fold's mean and population standard deviation for both features
and targets. Constant columns use a scale of one.

### Define a dataset

For a reusable numeric table, give a `CsvDataset` subclass its filename and
target column. The returned object is an instance of that subclass.

```python
from mgs_kan.data import CsvDataset, register_dataset

class HousingDataset(CsvDataset):
    filename = "houses.csv"
    target = "price"

data = HousingDataset.load(root="data", fold=0)
register_dataset("housing", HousingDataset)  # Optional name for load_dataset("housing").
```

For CSV-specific validation, transforms or grouped folds, override the class
method `prepare_columns(cls, header, values)`. For a different file format,
subclass `TabularDataset` and implement the class method `read(cls, path)`.
Both hooks return three arrays in matching row order: raw features of shape
`[N, D]`, targets of shape `[N]` and group labels of shape `[N]`. The base
class keeps each group in a single split and fits scaling on training rows.
To support `download=True`, set the class attribute `url`.

`ConcreteDataset` demonstrates the CSV hook for mixture recipes.
`ConcreteDataset.load(...)` and `load_dataset("concrete", ...)` return the
same dataset type. Name registrations last for the current Python process.

## Repository layout

- `src/mgs_kan/` contains the scalar model factory and its original settings.
- `tests/` covers both scalar models, dataset preprocessing and offline download behavior.
- `archive/` retains the previous working directory layout. It includes four
  notebooks, both scalar study protocols, geometry and interaction-order
  benchmarks, the report and its figures. See the [archive guide](archive/README.md).
- `requirements.txt` and `requirements-study.txt` retain the historical environment pins.
- `data/` and `results/` are ignored locations for future local inputs and outputs.
