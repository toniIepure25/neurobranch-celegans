"""Export dashboard-friendly covariance, autocorrelation and example artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.joint_dynamics.evaluation import autocorrelation_profile, pooled_neural_covariance

OUT = Path("reports/simulator/joint_dynamics")
MODELS = {
    "J0_persistence": ("persistence", "probabilistic persistence, block-bootstrap innovations", "baseline"),
    "J1_independent_diag": ("independent AR", "per-neuron AR(1), diagonal Gaussian lead-1 innovations", "baseline"),
    "J2_population_diag": ("population VAR diag", "rolling population VAR(1), diagonal Gaussian lead-1 innovations", "population mean + marginal noise"),
    "J3_population_full": ("population VAR full", "rolling population VAR(1), Ledoit-Wolf full covariance", "covariance test"),
    "J4_independent_full": ("independent AR full", "per-neuron AR(1) mean, full covariance from its residuals", "factorial covariance control"),
    "J5_population_lead1_boot": ("population lead-1", "rolling population VAR(1), lead-1 block bootstrap", "corrected historical comparator"),
}


def main() -> None:
    z = np.load(OUT / "trajectory_examples.npz")
    truth_keys = sorted(k for k in z.files if k.startswith("truth_"))
    model_keys = sorted(k for k in z.files if not k.startswith("truth_"))
    cov_rows = []
    if not (OUT / "covariance_h20.csv").exists():
        for label, keys in [("observed", truth_keys)] + [(m, [k for k in model_keys if k.endswith(m)]) for m in MODELS]:
            pooled = pooled_neural_covariance(np.concatenate([z[k] for k in keys], axis=0))
            corr = pooled / np.sqrt(np.outer(np.diag(pooled), np.diag(pooled)))
            for i in range(corr.shape[0]):
                for j in range(corr.shape[1]):
                    cov_rows.append({"model": label, "row": i, "col": j, "correlation": float(corr[i, j])})
        pd.DataFrame(cov_rows).to_csv(OUT / "covariance_h20.csv", index=False)
    acf_rows = []
    if not (OUT / "autocorr_profile.csv").exists():
        for label, keys in [("observed", truth_keys)] + [(m, [k for k in model_keys if k.endswith(m)]) for m in MODELS]:
            profile = autocorrelation_profile(np.concatenate([z[k] for k in keys], axis=0))
            for lag, value in enumerate(profile, start=1):
                acf_rows.append({"model": label, "lag": lag, "autocorrelation": float(value)})
        pd.DataFrame(acf_rows).to_csv(OUT / "autocorr_profile.csv", index=False)
    key = "0_2690"
    observed = z[f"truth_{key}"]
    example, bands = [], []
    for m in MODELS:
        ens = z[f"{key}_{m}"]
        for member in range(min(50, ens.shape[0])):
            for step in range(20):
                example.append({"model": m, "member": member, "step": step + 1, "pc1": float(ens[member, step, 0])})
        for step in range(20):
            bands.append(
                {
                    "model": m,
                    "step": step + 1,
                    "q05": float(np.quantile(ens[:, step, 0], 0.05)),
                    "q50": float(np.quantile(ens[:, step, 0], 0.50)),
                    "q95": float(np.quantile(ens[:, step, 0], 0.95)),
                }
            )
    pd.DataFrame(example).to_csv(OUT / "example_trajectories.csv", index=False)
    pd.DataFrame(bands).to_csv(OUT / "example_bands.csv", index=False)
    pd.DataFrame({"step": np.arange(1, 21), "observed_pc1": observed[:, 0], "observed_pc2": observed[:, 1]}).to_csv(
        OUT / "example_observed.csv", index=False
    )
    (OUT / "model_cards.json").write_text(
        json.dumps(
            {m: {"label": v[0], "definition": v[1], "role": v[2]} for m, v in MODELS.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("exported covariance, autocorrelation and examples")


if __name__ == "__main__":
    main()
