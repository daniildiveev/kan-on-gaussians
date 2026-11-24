# Research archive

Use this directory as the working directory for historical commands. The
notebooks import `src` and `experiments` from here. The geometry benchmark reads
selected cells from the original notebook. The study protocols fingerprint
source files, so their names and contents matter for replay.

| Research | Files |
| --- | --- |
| Scalar MGS-KAN and MLP, expanded study | `study.ipynb`, `src/`, `experiments/pair_ablation.py`, `experiments/pair_confirmation.py` |
| Earlier scalar protocol with analytic interaction scores | `mgs_kan_study.ipynb`, `experiments/mgs_kan/` |
| Gaussian-state spline edges | `gaussian_spline_kan_v1.ipynb` |
| Gaussian geometry and Bures–Wasserstein aggregation | `gaussian_kan_geometry_vs_mlp_v2.ipynb`, `experiments/geometry_{reference,benchmark,confirmation}.py`, `experiments/gaussian_layer_checks.{py,json}` |
| Interaction order, shifts and concrete strength | `experiments/{hypergraph_order,geometry_order,geometry_shift,planted_geometry,concrete_order}.py`, final sections of `study.ipynb` |
| Report and recorded results | `MGS-KAN-Experimental-Report.tex`, `study-math.tex`, `report-figures/`, saved notebook outputs and algebra-check JSON |

The model and config files under `src/` preserve the exact source bytes used for
historical study fingerprints. Archived models use their original constants.
The requirements files link to the pinned files at the repository root.

## Reproduce or inspect

From the repository root, in a dedicated Python 3.12 environment:

```sh
cd archive
python -m pip install -r requirements-study.txt
python -m experiments.mgs_kan.checks
python -m experiments.mgs_kan.experiment --help
python -m jupyterlab study.ipynb
```

The module check exercises the earlier scalar model without training. The CLI
runs that study when invoked without `--help`. Open notebooks to inspect saved
outputs. Executing training cells starts or resumes fits. Use a working copy to
keep the recorded outputs intact. The report's reproduction section lists stage
paths and commands relative to this directory.

For the first spline notebook, use a separate environment with
`requirements.txt`. The geometry comparison notebook also imports pandas and
records pandas 3.1.0 in its saved environment. Install it separately when
recreating that environment.

Historical checkpoints and run manifests under `outputs/study/` and
`gkan_results/main/` are absent from this checkout. Restore the original bundles
under those paths for stages that verify and reuse earlier fits. The concrete
benchmark expects the UCI Concrete Compressive Strength CSV at
`data/concrete_compressive_strength.csv` under this directory. Its source URL
and expected header are in `experiments/concrete_order.py`.

The scalar MLP and pairwise MGS-KAN are directly usable for 12-feature regression.
The geometry models consume Gaussian states, and the concrete models assume eight
specific inputs. Reuse on other tabular tasks requires adaptation. No independent
FastKAN or classical scalar spline-KAN baseline was found in this study.
