"""Validate the adaptive-dynamics report contract and artifact manifest."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neurobranch.adaptive.manifest import verify_manifest
from neurobranch.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "reports/simulator/adaptive_dynamics"
REQUIRED = [
    "model_paired_overall.csv",
    "model_paired_by_session.csv",
    "model_per_session_h20.csv",
    "model_per_neuron_h20.csv",
    "model_horizon_curves.csv",
    "model_update_summary.csv",
    "model_configurations.json",
    "model_cards.json",
    "nonstationarity_windows.csv",
    "nonstationarity_null.csv",
    "nonstationarity_summary.csv",
    "nonstationarity_synthetic.csv",
    "calcium_summary.csv",
    "calcium_forecasts.csv",
    "calcium_diagnostics.json",
    "adaptation_indicators.csv",
    "adaptation_regression.csv",
    "adaptation_strata.csv",
    "perturbation_sensitivity.csv",
    "oxygen_adaptive_summary.csv",
    "oxygen_adaptive_per_landmark.csv",
    "gates.json",
    "manifest.json",
]


def main() -> int:
    failures = []
    for name in REQUIRED:
        if not (OUT / name).exists():
            failures.append(f"missing {name}")
    if failures:
        print("\n".join(failures))
        return 1

    paired = pd.read_csv(OUT / "model_paired_overall.csv")
    required_cols = {"model", "mean_mse_improvement", "ci_low", "ci_high", "fraction_landmarks_better"}
    if not required_cols.issubset(paired.columns):
        failures.append("paired table missing columns")
    if not np.isfinite(paired[["mean_mse_improvement", "ci_low", "ci_high"]].to_numpy()).all():
        failures.append("paired table contains non-finite values")
    if not ((paired.ci_low <= paired.mean_mse_improvement) & (paired.mean_mse_improvement <= paired.ci_high)).all():
        failures.append("paired CI does not contain the point estimate")

    sessions = pd.read_csv(OUT / "model_per_session_h20.csv")
    if sorted(sessions.worm_id.unique()) != [0, 1, 2, 3, 4]:
        failures.append("per-session table does not cover all five sessions")
    if not sessions.mean_mse.map(np.isfinite).all():
        failures.append("per-session MSE contains non-finite values")

    calcium = pd.read_csv(OUT / "calcium_summary.csv")
    corrected = calcium[calcium.model == "corrected_calcium"]
    if corrected.empty or not corrected.finite_fraction.eq(1.0).all():
        failures.append("corrected calcium model has non-finite trajectories")

    gates = json.loads((OUT / "gates.json").read_text(encoding="utf-8"))
    if gates["gates"]["D_distributional_improvement"]["status"] != "FAIL":
        failures.append("distributional gate was not recorded as failed")
    if gates["gates"]["F_independent_biological_validation"]["status"] != "NOT_EVALUABLE":
        failures.append("independent validation gate was not recorded as not evaluable")

    ok, problems = verify_manifest(OUT)
    if not ok:
        failures.extend(problems)

    if failures:
        print("FAILED")
        print("\n".join(failures))
        return 1
    print(f"validated {len(REQUIRED)} required artifacts; manifest verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
