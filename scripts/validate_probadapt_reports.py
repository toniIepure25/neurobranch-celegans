"""Validate the probabilistic adaptive-dynamics report contract and manifest."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neurobranch.adaptive.manifest import verify_manifest
from neurobranch.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "reports/simulator/probabilistic_adaptive"
REQUIRED = [
    "reproduction_landmarks.csv",
    "reproduction_summary.csv",
    "block_length_sensitivity.csv",
    "reproduction_by_session.csv",
    "forecast_overlap.csv",
    "reproduction_metadata.json",
    "baseline_validation.csv",
    "baseline_per_landmark.csv",
    "baseline_per_neuron_h20.csv",
    "baseline_updates.csv",
    "baseline_per_session_h20.csv",
    "baseline_horizon.csv",
    "baseline_paired_vs_frozen.csv",
    "baseline_paired_vs_local_mean.csv",
    "baseline_paired_vs_independent_ar.csv",
    "baseline_paired_by_session.csv",
    "baseline_metadata.json",
    "residual_diagnostics.csv",
    "residual_covariance.csv",
    "residual_autocorrelation.csv",
    "residual_covariance_sensitivity.csv",
    "residual_summary.csv",
    "innovation_choice.json",
    "prob_per_landmark.csv",
    "prob_per_neuron_h20.csv",
    "prob_updates.csv",
    "prob_variance_decomposition.csv",
    "prob_summary.csv",
    "prob_paired.csv",
    "prob_paired_all.csv",
    "prob_per_session.csv",
    "prob_horizon.csv",
    "prob_variance_summary.csv",
    "prob_pit.csv",
    "prob_configurations.json",
    "prob_model_cards.json",
    "prob_metadata.json",
    "prob_example_trajectories.csv",
    "prob_example_bands.csv",
    "prob_example_diagnostics.csv",
    "prob_example_metadata.json",
    "prob_gates.json",
    "prob_perturbation_status.json",
    "figures/prob_energy_comparison.png",
    "figures/prob_calibration.png",
    "figures/prob_variance_decomposition.png",
    "figures/prob_horizon.png",
    "figures/prob_example_rollouts.png",
    "figures/prob_residual_autocorrelation.png",
    "figures/figure_manifest.json",
    "manifest.json",
]


def _check_finite(frame: pd.DataFrame, columns: list[str], label: str, failures: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            failures.append(f"{label}: missing column {column}")
        elif not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            failures.append(f"{label}: non-finite values in {column}")


def main() -> int:
    failures = [f"missing {name}" for name in REQUIRED if not (OUT / name).exists()]
    if failures:
        print("FAILED")
        print("\n".join(failures))
        return 1

    paired = pd.read_csv(OUT / "prob_paired.csv")
    _check_finite(paired, ["mean_improvement", "ci_low", "ci_high"], "prob_paired", failures)
    if not ((paired.ci_low <= paired.mean_improvement) & (paired.mean_improvement <= paired.ci_high)).all():
        failures.append("prob_paired: CI does not contain point estimate")

    summary = pd.read_csv(OUT / "prob_summary.csv")
    _check_finite(summary, ["mean", "median"], "prob_summary", failures)
    coverage = summary[summary.metric.isin(["coverage_50", "coverage_80", "coverage_95"])]
    if not coverage["mean"].between(0, 1).all():
        failures.append("prob_summary: coverage outside [0, 1]")

    session = pd.read_csv(OUT / "prob_per_session.csv")
    if sorted(session.worm_id.unique()) != [0, 1, 2, 3, 4]:
        failures.append("prob_per_session: does not cover all five sessions")

    landmark = pd.read_csv(OUT / "prob_per_landmark.csv")
    if landmark[["worm_id", "landmark"]].drop_duplicates().shape[0] != 175:
        failures.append("prob_per_landmark: expected 175 unique landmarks")
    _check_finite(landmark, ["energy_h20", "crps_h20_mean", "spread_error_ratio_h20"], "prob_per_landmark", failures)

    variance = pd.read_csv(OUT / "prob_variance_summary.csv")
    if not (variance["within_ensemble_var"] >= 0).all():
        failures.append("prob_variance_summary: negative variance")

    gates = json.loads((OUT / "prob_gates.json").read_text(encoding="utf-8"))
    if gates["gates"]["D_probabilistic_score_advantage"]["status"] != "FAIL":
        failures.append("Gate D was not recorded as failed")
    if gates["gates"]["E_calibration"]["status"] != "FAIL":
        failures.append("Gate E was not recorded as failed")
    if gates["gates"]["F_trajectory_fidelity"]["status"] != "FAIL":
        failures.append("Gate F was not recorded as failed")
    if gates["gates"]["H_biological_validation"]["status"] != "NOT_EVALUABLE":
        failures.append("Gate H was not recorded as not evaluable")

    perturbation = json.loads((OUT / "prob_perturbation_status.json").read_text(encoding="utf-8"))
    if perturbation.get("status") != "NOT_LAUNCHED":
        failures.append("perturbation status is not NOT_LAUNCHED")
    if perturbation.get("warning") != "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.":
        failures.append("perturbation warning text is not exact")

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
