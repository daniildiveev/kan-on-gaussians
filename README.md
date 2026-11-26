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

When upgrading an existing notebook environment, reinstall the editable project and
restart the kernel. Import project modules through `src`, for example
`from src.experiments import ExperimentRunner`. The pip distribution name is `mgs-kan`.
Dataset roots and local checkpoint paths are resolved from the checkout. Absolute paths can point elsewhere.

Use Python 3.11+ in a project environment. Install the test extra to run pytest.
For historical notebook environments, see the archive guide.

```sh
python -m pip install -e '.[test]'
python -m pytest
```

```python
import torch
from src.config import ModelSettings
from src.models import make_model

model = make_model("mgs_kan", ModelSettings(features=8))
with torch.no_grad():
    predictions = model(torch.zeros(2, 8))
```

`make_model("mgs_kan")` reads settings from the environment at construction.
For example, `MGS_KAN_FEATURES=8` selects eight inputs for either model.
Explicit field values override environment variables. Defaults reproduce the study models.

## Load regression data

Pass a registry name to `load_dataset` or set `DATASET` in the baseline notebook.
Use `download=True` once to populate the local cache from the linked sources.
`dataset_class(name).available_folds` lists the supported folds before loading data.
The same train-only standardization and inverse target transform apply to every split.

| Registry name | Inputs | Target | Test protocol and source |
| --- | ---: | --- | --- |
| `airfoil` | 5 | Scaled sound pressure (dB) | Five folds grouped by all four non-frequency conditions of the wind tunnel experiment. [NASA / UCI](https://doi.org/10.24432/C5VW2C) |
| `power_plant` | 4 | Net power (MW) | Five folds grouping identical inputs. The UCI CSV contains one permutation of the 9,568 observations. [UCI](https://doi.org/10.24432/C5002N) |
| `sarcos` | 21 | First joint torque | Fold 0 uses the 4,449 official test rows. A seeded 20% of the 44,484 development rows is reserved for validation. The cache bundles both original MAT files. [GPML](https://gaussianprocess.org/gpml/data/) |
| `gas_turbine_co`, `gas_turbine_nox` | 9 | CO or NOx (mg/m³) | Fold 0 trains on 2011–12, validates on 2013 and tests on 2014–15. Both tasks share the same nine operating measurements and cache. [UCI](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set) |
| `superconductivity` | 81 | Critical temperature (K) | Five folds grouped by elemental fractions, rounded to 12 decimals, from the matching composition table. [UCI](https://archive.ics.uci.edu/dataset/464/superconductivty+data) |
| `appliances` | 26 | Appliance energy (Wh per ten-minute interval) | Fold 0 uses the earliest 70% for training, next 15% for validation and last 15% for testing. Inputs are 18 room measurements, six weather variables and sine/cosine of time of day. Lighting usage and the two random controls are excluded. [UCI](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) |
| `concrete` | 8 | Compressive strength (MPa) | Five folds grouped by mixture recipe. [UCI](https://doi.org/10.24432/C5PK67) |
| `wine_quality` | 11 | Sensory quality score | Red and white wines pooled. Five folds group identical chemical measurements across both files. [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) |

```python
import torch
from torch.utils.data import DataLoader
from src.config import ModelSettings
from src.data import CsvSpec, load_dataset
from src.models import make_model

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
from src.data import CsvDataset, register_dataset

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
from src.config import ModelSettings, TrainingSettings
from src.data import load_dataset
from src.models import make_model
from src.training import Trainer

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

Use `TrainingSettings(_env_file=ENV_FILE)` to load the `MGS_KAN_TRAIN_*` values
from the repository dotenv file. Import `ENV_FILE` from `src.paths`.

### Track runs from Jupyter

Install the client in the notebook environment with `%pip install -e '.[tracking]'`
from the repository root. Set `MLFLOW_TRACKING_URI` to the existing server URL
in the kernel environment or in `.env`. Set `MLFLOW_EXPERIMENT_NAME` to select
an experiment (default `mgs-kan`). The client uses standard MLflow credentials
from the kernel environment if the server requires authentication.

```python
from src.config import TrackingSettings
from src.paths import ENV_FILE
from src.tracking import MlflowTracker

tracker = MlflowTracker(TrackingSettings(_env_file=ENV_FILE))
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

### Run a fold and seed series from Jupyter

For a validation survey across every benchmark, open
[the two-wave notebook](notebooks/baselines/benchmark_survey.ipynb).
`ValidationGrid.preset("benchmark_survey").wave(1)` and `.wave(2)` each
schedule 36 combinations: nine regression tasks, four families, all inputs,
fold 0, seed pair (11, 101). Each wave writes its run IDs and per-task
validation RMSE, R² at the best epoch, parameter counts and training times to
`results/benchmark-survey/<VERSION>/` before the next wave starts.
Superconductivity uses one worker at a time under the same GPU memory budget.

The survey searches the configured MLflow experiment across series names.
Reuse requires a finished, evaluated validation run with its checkpoint and
matching source data hash, target, fold, ordered inputs, seeds, model-specific
settings, Adam options, batch size, epoch count and device backend. For old
runs the dataset reader's fold convention is assumed unchanged. For several
matches the newest qualifying run is used. Historical training times include
the original run's hardware and load.

For a size-matched comparison on Airfoil and Power Plant, open the
[parameter budget notebook](notebooks/baselines/parameter_budget.ipynb).
`ValidationGrid.preset("parameter_budget")` compares MLP, MGS-KAN, deep MLP
and ResNet at 1,000, 4,000 and 16,000 trainable parameters. On fold 0 it
requires 36 compact runs from MLflow and schedules up to 72 new runs.
The notebook plots validation RMSE against actual parameter counts and
recorded run times. The timing plot uses complete seed groups with matching
GPU model, CPU thread count and single-worker execution.

For architecture selection, open [the validation notebook](notebooks/baselines/select_models.ipynb).
Its Pydantic grid definitions live in [search.py](src/experiments/search.py).
These profiles use fold 0, three paired initialization/shuffle seeds, Adam,
a batch size of 64, a learning rate of 0.001 and 100 epochs. They rank candidates
using the validation RMSE at the saved best epoch. Test scoring is deferred.

| Profile | Datasets and candidates | Fits |
| --- | --- | ---: |
| `airfoil_first` (default) | Airfoil, MLP/MGS-KAN stages 1 and 2 | 12 |
| `airfoil_all` | Airfoil, the same candidates plus stage 3 | 18, including the original 12 |
| `deep` | All nine registered regression tasks, four-layer MLP and four-block tabular ResNet, both width 128 | 54 |
| `features` | Dataset-specific feature subsets, three configurations of each of four model families | 1,656 for Airfoil + Power Plant |

For the feature search set `PROFILE = "features"` and
`FEATURE_DATASETS = ("airfoil", "power_plant")` in the notebook. Use a one-element tuple
to run either dataset alone, or `None` for all nine tasks (2,916 fits). Inspect the preview
before executing the training cell. Each series resumes through MLflow under its dataset,
stage, feature-set identifier and version. MLflow records the ordered feature names,
input count and actual parameter count.

Airfoil and Power Plant enumerate all 31 and 15 nonempty subsets. The other policies
are defined in [features.py](src/experiments/features.py). SARCOS combines the three
joint measurement groups in all seven ways. The other tasks pair the full set with
physics or chemistry groups. All candidates within a fold use the same row partitions
and train-only scaling. Projection happens after the dataset's usual preprocessing.

In this profile the two-hidden-layer MLP takes the integer width closest in parameter
count to its paired MGS-KAN for the chosen input dimension. Deep MLP candidates have
3×64, 4×128 and 6×256 hidden layers × width. The ResNet candidates use 2×64, 4×128 and
6×256 residual blocks × width. The notebook reports validation winners by feature subset
and family, dispersion across subsets of each size and paired MLP minus MGS-KAN RMSE.

Airfoil's stages use MLP widths 7, 10 and 15 with two hidden layers. The
corresponding MGS-KAN kernels per feature/pair are 3/1, 4/2 and 7/4.
The matched parameter counts are 106/111, 181/186 and 346/351.
Promote a larger candidate within a family when mean validation RMSE improves
by at least 1% and a majority of paired seeds improves. Compare results within
each dataset and outer fold. The deep profile measures achievable accuracy at
a larger budget. Its residual blocks use pre-layer normalization and two linear
layers with a ReLU in the residual branch.

Select a visible GPU in the notebook. Keep `VERSION` to resume completed runs in
the same MLflow experiment after a code update. Choose a new version when you
change the data, model, or training settings and need fresh results.
Each dataset and stage has its own series ID.
Each call to `grid.run()` uses one heartbeat session. Telegram announces the start,
the first newly trained run with its validation RMSE, progress every 10% from
10% through 90%, and completion. If one run crosses several thresholds, only
the highest is announced. Counts include reusable MLflow runs across the grid.
Resuming skips thresholds already reached. Thresholds reached on the final run
are covered by the completion summary. Elapsed time starts with this call.
Failures and resource alerts identify the
active series. Update the heartbeat receiver in `vps-metrics` first.

Each `grid.run()` or standalone `ExperimentRunner.run()` prints the absolute
path to a new text log under `results/logs/` in the project root. Use
`tail -f /path/to/the/printed.log` to follow it during training. UTC entries
record reused MLflow runs, trial seeds, best validation RMSE, elapsed time,
progress and failure tracebacks. Set `MGS_KAN_LOG_DIR` in the root `.env`
to choose another directory. Resuming the same version creates a new file.

Open [the baseline series notebook](notebooks/baselines/run_series.ipynb).
In the kernel environment install `python -m pip install -e '.[tracking]' pandas` from the checkout.
The first cells locate the checkout from the notebook directory or the editable install.
For a kernel started elsewhere, you can set `PROJECT_DIR` to the absolute checkout path
on the server. Check the displayed project, dotenv and data paths, then choose your
visible GPU and run `smoke` or `full`. Configure the MLflow URI in the kernel or root dotenv file.
The notebook defaults to Airfoil. Full mode runs both models with three seed pairs
on every supported fold (30 runs for five folds, six for a single fixed test).
Cache the dataset under `root`, or pass `download=True` to allow the initial download.

```python
from src.config import ResourceSettings, TrackingSettings, TrainingSettings
from src.paths import ENV_FILE
from src.experiments import ExperimentRunner

runner = ExperimentRunner(
    training=TrainingSettings(epochs=50, batch_size=64),
    resources=ResourceSettings(_env_file=ENV_FILE),
    tracking=TrackingSettings(_env_file=ENV_FILE),
)
summary = runner.run(
    series_id="concrete-baselines-v1",
    dataset="concrete",
    root="data",
    models=["mlp", "mgs_kan"],
    folds=[0, 1, 2, 3, 4],
    seeds=[(11, 101), (22, 202), (33, 303)],
    device="cuda:1",
    max_runs=30,
)
summary.by_model
```

Each seed pair gives the model initialization and batch shuffle seeds. Each
combination starts a fresh Python process, trains with Adam, restores the
best validation epoch and evaluates the test fold. Use `device="cpu"` for
a local test. CUDA indices follow the kernel's existing visible-device order.
Set `MGS_KAN_SERIES_CPU_THREADS` (default 2) and
`MGS_KAN_SERIES_GPU_MEMORY_FRACTION` (default 0.25) to limit worker threads
and the CUDA allocator. Other users can consume memory on the same GPU.

[Parallel Power Plant notebook](notebooks/baselines/parallel_power_plant.ipynb)
runs MLP, MGS-KAN, Deep MLP and ResNet on the manually selected GPU. All four
start on fold 0 with the same seed pair. The queue then fills available slots
with the next pending trials. Each process has one CPU thread in this profile.
Set `max_parallel_trials` from 1 (default) to 4 in `ResourceSettings`.
Scheduling is bounded within each series. Open a single notebook per series ID.

On CUDA, parallel series share a default 4096 MiB budget. Set `gpu_budget_mib`
to override it or enable measurement for a sequential comparison. When worker
PIDs are visible in `nvidia-smi`, each worker subtracts its CUDA context from
its allocator share and the runner samples the combined usage of those PIDs.
If Jupyter hides a worker PID, the worker resolves the selected CUDA device's
UUID and the runner monitors total memory on that card, including other users.
In this mode each worker limits its allocator to its share of the remaining
budget. The runner stops the series on an observed overrun. Short peaks between
samples are possible. The notebook reports worker and whole-card peaks in
separate columns. Interrupting the notebook stops the active workers.

`max_runs` caps the full grid. `time_budget_seconds` lets a running trial
finish and defers the next one. Errors stop the series and identify the
combination. Repeat the call with the same series ID and MLflow experiment to
reuse completed evaluations matching the requested model, fold, seed pair and
evaluation mode. Run one controller per series ID. Choose a new series ID for
experiments you want to compare separately. MLflow keeps the protocol and
source hashes for each attempt.
MLflow records each attempt under the tag `mgs_kan.series`.

`summary.results` contains individual run IDs and scores, and `summary.pending`
contains the remaining combinations. `summary.by_model` reports coverage
and averages the available seed repeats within each fold before averaging
across folds. R² uses only folds with defined scores.

## Repository layout

- `src/config.py` contains Pydantic settings for models, training, tracking and resources.
- `src/data/` loads and splits tabular data. `src/models/` contains the model classes and factory.
- `src/training/` owns single-model training and checkpoints. `src/tracking/` owns MLflow runs and history.
- `src/experiments/` defines series, workers, the notebook runner and result summaries.
- `tests/` covers both scalar models, dataset preprocessing and offline download behavior.
- `archive/` retains the previous working directory layout. It includes four
  notebooks, both scalar study protocols, geometry and interaction-order
  benchmarks, the report and its figures. See the [archive guide](archive/README.md).
- `requirements.txt` and `requirements-study.txt` retain the historical environment pins.
- `data/` and `results/` are ignored locations for future local inputs and outputs.

## Optional series alerts

`ExperimentRunner` can send progress to the heartbeat service maintained in
`vps-metrics`. Set `MGS_KAN_MONITOR_ENDPOINT=https://<alerts-host>/v1/events`
and `MGS_KAN_MONITOR_TOKEN` to the receiver bearer token in the training
environment. For a local dotenv file use the same runner API:

```python
from src.config import MonitorSettings
from src.paths import ENV_FILE
from src.experiments import ExperimentRunner

runner = ExperimentRunner(monitoring=MonitorSettings(_env_file=ENV_FILE))
```

Registration starts before the dataset probe. Heartbeats continue every minute
during training. If delivery fails, the notebook warns and training continues.
For disk warnings, set `MGS_KAN_MONITOR_DISK_PATH` and
`MGS_KAN_MONITOR_DISK_THRESHOLD_GIB` on the training machine.
