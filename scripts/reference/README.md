# Julia reference run

This directory contains the standalone extraction of the authors' `simulations.ipynb`
cells used to check the Python clamped-core port. It is not a second model.

## Prerequisites

- Julia 1.11.2 (the release's tested version)
- The third-party `Celegans_premotor` release restored under
  `references/external/Celegans_premotor` by `python scripts/fetch_vienna_artifacts.py`
- A writable Julia depot. The package set is installed with:

```bash
export JULIA_DEPOT_PATH=/path/to/julia-depot
julia --startup-file=no scripts/reference/install_julia_pkgs.jl
```

`install_julia_pkgs.jl` installs `OrdinaryDiffEq`, `JSON`, `JLD2`, `DataFrames`, `CSV`,
`SavitzkyGolay` and `DataInterpolations`.

## Run

```bash
export JULIA_DEPOT_PATH=/path/to/julia-depot
julia --startup-file=no scripts/reference/run_julia_reference.jl 2023-01-23-15
```

The script writes `reports/vienna_workbench/julia_reference/<recording>_julia.csv` with
columns `time`, `truth_<neuron>` and `sim_<neuron>` for the 15 intrinsic neurons. It also
writes `_A.csv`, `_W.csv`, `_core.csv` and `_ut0.csv` diagnostics; these are not required
for the product and can be deleted.

## Compare

```bash
PYTHONPATH=src python scripts/compare_tier3_reference.py
```

The comparison reports per-neuron Pearson r, maximum absolute difference, RMSE, input
SHA-256 hashes, Git state and the exact command. The canonical result is mean
r=0.9999997 and max |Δ|=0.0133. That is a port check, not a biological validation.

The Julia run uses `Tsit5`; the Python port uses `RK45` with `max_step=dt`. The small
residual difference is solver tolerance, not a parameter mismatch.
