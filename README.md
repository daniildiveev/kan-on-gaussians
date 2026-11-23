# MGS-KAN

This repository studies Multidimensional Gaussian Surface Kolmogorov–Arnold Networks.
The active package exposes two baselines from the scalar study. By default, both
accept batches of 12 features and return one prediction per example.

| Factory name | Model | Trainable parameters |
| --- | --- | ---: |
| `mlp` | Two ReLU hidden layers of width 24 | 937 |
| `mgs_kan` | Affine term plus single-feature and pairwise Gaussian kernels | 949 |

The next research stage is comparison on real datasets. The active package
currently provides model construction and inference. The published study
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

## Repository layout

- `src/mgs_kan/` contains the scalar model factory and its original settings.
- `tests/` contains CPU smoke tests for both models.
- `archive/` retains the previous working directory layout. It includes four
  notebooks, both scalar study protocols, geometry and interaction-order
  benchmarks, the report and its figures. See the [archive guide](archive/README.md).
- `requirements.txt` and `requirements-study.txt` retain the historical environment pins.
- `data/` and `results/` are ignored locations for future local inputs and outputs.
