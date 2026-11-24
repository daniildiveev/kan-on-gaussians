# Working in this repository

Discuss each proposed change with the user before implementing it. Inspect the
current checkout, explain the scope, identify affected entry points, and describe
how to verify the result. Wait for the user's explicit approval of the approach.
The user manages commits and pull requests unless they explicitly delegate them.

## Research code

- Use Pydantic Settings for active model configuration and pytest for unit tests.
- Import active code from `config`, `data`, `models`, `training`, `tracking`, and
  `experiments`. Keep training a single model in `Trainer`, scheduling a series
  in `ExperimentRunner`, and MLflow history queries in `MlflowTracker`.
- Check Git status before editing. Preserve user changes, recorded notebook
  outputs, figures, datasets, and trained weights.
- Trace imports and paths from notebooks, scripts, and reports before moving a
  file. Some study protocols hash Python source files or specific notebook cells.
  Treat those inputs and the original working directory as part of reproduction.
- For structural moves, use `git mv` for tracked files when applicable. Keep
  model equations, public interfaces, and dependency pins intact. Keep distinct
  historical implementations with their respective protocols.
- Verify moves with cheap before-and-after checks such as imports, fixed-seed
  forward passes, state-dict loading, and artifact hashes. Run checks that write
  results in a temporary directory. Document missing inputs for historical runs.
- Keep new architectures, training runs, and dataset downloads outside this
  structural refactor.
