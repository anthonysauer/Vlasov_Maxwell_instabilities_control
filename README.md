# Control of plasma instabilities: reproducibility package

This repository is the code package for the submitted manuscript **Control of plasma instabilities**. It contains one reproducibility notebook and a small Python package.

## What this package does

Run `reproduce_all_results.ipynb` from top to bottom to create:

- `outputs/figures/regenerated/`: figures regenerated from compact numerical arrays included in this repository.
- `outputs/figures/submitted_reference/`: exact PNG figures used in the submitted manuscript, copied from `data/reference_figures/` for visual comparison.

The notebook covers both experiments in the paper:

1. **Weibel instability, linear analytic control**
   - dispersion-function landscape,
   - equilibrium and field-energy histories,
   - Fourier-mode suppression,
   - kinetic-energy objective landscapes,
   - sensitivity to imperfect cancellation.
2. **Two-stream instability, nonlinear numerical control**
   - two-stream equilibrium,
   - optimization loss history,
   - optimized five-harmonic control coefficients,
   - selected optimization landscapes,
   - submitted final comparison figures.

The default notebook path is intentionally fast and deterministic. It uses curated, compact solver outputs stored in `data/precomputed/`; these are the numerical arrays needed for plotting the paper results. It does **not** rerun the manuscript-scale JAX solver or the full 20,000-step optimization by default. The full raw working tree was hundreds of megabytes because it also stored large intermediate distribution histories, checkpoints, image duplicates, and exploratory notebooks.

## Repository layout

```text
.
├── reproduce_all_results.ipynb      # single main notebook; run this first
├── vm_control/                      # minimal reusable Python package
│   ├── solver.py                    # JAX SAV-based Vlasov--Maxwell solver
│   ├── experiments.py               # initial-condition builders and optional solver runners
│   ├── profiles.py                  # equilibria and control-field parameterization
│   ├── dispersion.py                # Weibel D2 dispersion utilities
│   ├── plotting.py                  # all compact-data figure generation
│   └── data.py                      # data-path helpers
├── data/
│   ├── precomputed/                 # compact numerical arrays used by the notebook
│   ├── reference_figures/           # submitted-version manuscript PNGs
│   └── paper_metadata.json          # machine-readable experiment settings
├── scripts/
│   └── reproduce_all_figures.py     # command-line equivalent of the notebook plotting step
├── outputs/                         # generated locally; ignored by git except .gitkeep
└── requirements.txt
```

## Installation

Create a clean Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The package declares JAX because `vm_control/solver.py` and the optional solver smoke test use it. The default figure-regeneration path primarily uses NumPy/Matplotlib and precomputed compact arrays. SciPy is used by optional analytic dispersion recalculation utilities. For GPU/TPU runs of the full solver, install the matching JAX build for your system before running the expensive cells.

## Reproducing the manuscript outputs

### Notebook path

```bash
jupyter lab reproduce_all_results.ipynb
```

Then run all cells. The final cell asserts that at least 19 regenerated figures and 41 submitted reference figures have been created. For a headless check, run:

```bash
jupyter nbconvert --to notebook --execute reproduce_all_results.ipynb \
  --output executed_reproduce_all_results.ipynb
```

### Command-line plotting path

```bash
python scripts/reproduce_all_figures.py
```

This creates the same figure directories as the notebook plotting step.

## Validation performed for this release

This cleaned package was tested by unpacking the zip in a fresh directory and running:

```bash
python scripts/reproduce_all_figures.py
jupyter nbconvert --to notebook --execute reproduce_all_results.ipynb \
  --output executed_reproduce_all_results.ipynb
```

The expected successful run creates 19 regenerated PNG figures and copies 41 submitted-reference PNG figures.

## Optional full-solver reruns

The JAX solver and initial-condition builders are included in `vm_control/solver.py` and `vm_control/experiments.py`. The notebook contains an optional smoke-test cell that exercises the solver on a tiny grid. It is off by default:

```python
RUN_SOLVER_SMOKE_TEST = False
```

Change it to `True` to verify the solver path. For manuscript-scale reruns, use the paper-scale parameters below and expect substantial runtime and memory use.

### Weibel experiment parameters

| parameter | value |
|---|---:|
| thermal velocity `v_th` | 0.3 |
| temperature ratio `T_r` | 12 |
| fundamental wave number `k0` | 1.25 |
| perturbed Fourier mode | 3 |
| density perturbation amplitude `alpha_3` | 1e-3 |
| plasma magnetic amplitude | -1e-3 |
| ideal exterior magnetic amplitude | +1e-3 |

In code, the net magnetic amplitude passed to `run_weibel_simulation(...)` is:

- `net_B_amplitude=1e-3` for the no-control baseline,
- `net_B_amplitude=0.0` for exact cancellation,
- `net_B_amplitude=delta` for the mismatch tests.

### Two-stream experiment parameters

| parameter | value |
|---|---:|
| thermal velocity `v_th` | 0.2 |
| stream velocity `v_bar` | 0.7 |
| fundamental wave number `k0` / `beta` | 0.5 |
| density perturbation amplitude `alpha` | 1e-3 |
| final time `T_end` | 30 |
| controlled modes | 1, 2, 3, 4, 5 |

The optimized coefficients are available as:

```python
from vm_control.profiles import two_stream_best_params
params = two_stream_best_params()  # shape (5, 4), columns a_k,b_k,c_k,d_k
```
