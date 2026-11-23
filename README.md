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

## Train and evaluate a baseline

```python
import torch
from mgs_kan.config import ModelSettings, TrainingSettings
from mgs_kan.data import load_dataset
from mgs_kan.models import make_model
from mgs_kan.training import Trainer

data = load_dataset("concrete", fold=0, download=True)
settings = TrainingSettings(device="auto", seed=42, epochs=50, batch_size=64)
model = make_model("mlp", ModelSettings(features=data.n_features, init_seed=17))

trainer = Trainer(model, settings)
result = trainer.fit(data)
test_rmse = trainer.evaluate(data)
print(f"Best epoch {result.best_epoch}, test RMSE {test_rmse:.3f} MPa")

features, _ = next(iter(torch.utils.data.DataLoader(data.test, batch_size=64)))
predictions_mpa = data.inverse_target(trainer.predict(features))
```

Use `"mgs_kan"` in the same example for the pairwise baseline. By default each
`fit` creates Adam with `settings.learning_rate`. To use RMSprop, build it after
the trainer places the model on the training device:

```python
optimizer = torch.optim.RMSprop(trainer.model.parameters(), lr=0.002)
result = trainer.fit(data, optimizer=optimizer)
```

A supplied optimizer keeps its learning rates, parameter groups, and accumulated
state. The training loop optimizes standardized MSE. `ModelSettings.init_seed`
controls initial weights for both baselines and preserves the caller's RNG state.
Omitting it uses the current PyTorch RNG. `TrainingSettings.seed` controls batch
order independently.
Each epoch records the mean training loss per example in `result.train_mse` and
validation RMSE in original units in `result.validation_rmse`. After the final
epoch, `fit` restores both the model and optimizer to the earliest epoch with minimum
validation RMSE. Pass the same optimizer to a later call to continue from that state.
`evaluate` scores those weights on the test split.

The trainer holds the supplied model. `predict` accepts a batch of standardized
features and returns standardized predictions on the model's device. With
`Trainer(model)`, standalone prediction and evaluation use the existing model
device. The first `fit` loads the default training settings. The function calls
`fit(model, data, settings, optimizer=optimizer)` and `evaluate(model, data)` also remain available.

Training defaults to CPU. Use `TrainingSettings(device="cuda")` for the current
CUDA GPU, `"cuda:1"` for the second visible CUDA device, or `"mps"` for Apple
Metal. With `"auto"`, selection proceeds in order of CUDA, MPS, then CPU.
An unavailable explicit device raises an error. The dataset stays in CPU memory.
`fit` moves the model and training batches to the selected device and leaves
the fitted model there. `evaluate` follows the model's device. On MPS, predictions
return to CPU for float64 RMSE calculation.

Use `TrainingSettings(_env_file=".env")` to load the `MGS_KAN_TRAIN_*` values
from the example environment file.

### Track runs from Jupyter

Install the client in the notebook environment with `%pip install -e '.[tracking]'`
from the repository root. Set `MLFLOW_TRACKING_URI` to the existing server URL
in the kernel environment or in `.env`. Set `MLFLOW_EXPERIMENT_NAME` to select
an experiment (default `mgs-kan`). The client uses standard MLflow credentials
from the kernel environment if the server requires authentication.

```python
from mgs_kan.config import TrackingSettings
from mgs_kan.tracking import MlflowTracker

tracker = MlflowTracker(TrackingSettings(_env_file=".env"))
trainer = Trainer(model, settings, tracker=tracker)
result = trainer.fit(data)
test_rmse = trainer.evaluate(data)
print(result.run_id)
```

Each `fit` creates its own run, including repeated calls from the same cell.
It records the data source, fold, model settings, explicit `model.init_seed`,
`shuffle_seed` and each optimizer group's starting options (for example
`optimizer.betas.0` for the first Adam group). The legacy `seed` also means
batch order. Each run tags the on-disk package Python sources with a SHA-256
digest. In a Git checkout, tags also include HEAD and whether package sources
have uncommitted changes. Restart the notebook kernel after editing imported code
to train with the recorded source version. Open the
experiment in the server UI to compare `train_mse` (standardized loss),
`val_rmse` (original units) and `val_r2` at each epoch. `best_epoch` identifies
the restored weights. Calling `evaluate` with the fitted dataset instance adds
`test_rmse` and `test_r2` to that run. R² uses the full split. For constant
targets it is 1 for exact predictions and 0 otherwise. Single-row splits omit R².

Passing a tracker requires a reachable server and propagates logging failures to
the notebook. Omit `tracker` for ordinary training without the MLflow dependency.

### Save and load the best epoch

After `fit`, save the best validation weights, optimizer state and epoch together.
Tracked runs also upload this snapshot. Set `MLFLOW_CHECKPOINT_FILE` to choose
the artifact filename (default `best.pt`). Each run records its filename for later loading.
With older clients such as MLflow 2.17, set `MLFLOW_TRACKING_URI` in the kernel
environment when using the server's artifact proxy.

```python
optimizer = torch.optim.RMSprop(trainer.model.parameters(), lr=0.002)
result = trainer.fit(data, optimizer)
trainer.save_checkpoint("results/best.pt")

# In a fresh notebook, construct the same model and optimizer parameter groups.
restored = Trainer(make_model("mlp", ModelSettings(features=data.n_features)), settings)
optimizer = torch.optim.RMSprop(restored.model.parameters())
epoch = restored.load_checkpoint("results/best.pt", optimizer=optimizer)
test_rmse = restored.evaluate(data)
continued = restored.fit(data, optimizer)
```

The example assumes the original run used RMSprop. Supply the same optimizer
class with parameters in the same groups and order. Loading restores group
settings, including learning rates. Omit `optimizer` to load weights for prediction.
Use the same feature and target preprocessing as in training.

For a tracked run, construct the destination trainer with `tracker=tracker` and
call `restored.load_checkpoint(run_id=result.run_id, optimizer=optimizer)`.
A subsequent `fit` starts a new run and a new epoch count from the loaded state.

## Repository layout

- `src/mgs_kan/` contains the scalar model factory and its original settings.
- `tests/` covers both scalar models, dataset preprocessing and offline download behavior.
- `archive/` retains the previous working directory layout. It includes four
  notebooks, both scalar study protocols, geometry and interaction-order
  benchmarks, the report and its figures. See the [archive guide](archive/README.md).
- `requirements.txt` and `requirements-study.txt` retain the historical environment pins.
- `data/` and `results/` are ignored locations for future local inputs and outputs.
