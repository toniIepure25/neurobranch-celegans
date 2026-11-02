"""Validate the uncertainty-calibration report contract, gates and manifest."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neurobranch.adaptive.manifest import sha256_file, verify_manifest
from neurobranch.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "reports/simulator/uncertainty_calibration"
REQUIRED = [
    "variance_propagation.csv",
    "variance_decomposition.csv",
    "variance_decomposition.json",
    "operator_diagnostics.csv",
    "operator_summary.csv",
    "innovation_validation.csv",
    "innovation_summary.csv",
    "lead_profile.csv",
    "synthetic_controls.csv",
    "synthetic_summary.json",
    "lowrank_validation.csv",
    "lowrank_decision.json",
    "d4_validation.csv",
    "d4_scores.csv",
    "d4_coverage.csv",
    "d4_trajectory.csv",
    "d4_summary.json",
    "reproduction_check.csv",
    "reproduction.json",
    "scores.csv",
    "score_summary.csv",
    "paired_scores.csv",
    "trajectory_scores.csv",
    "trajectory_paired.csv",
    "coverage_by_horizon.csv",
    "coverage_summary.csv",
    "variance_summary.csv",
    "covariance_h20.csv",
    "per_neuron_h20.csv",
    "per_session_calibration.csv",
    "calibration_h20.csv",
    "calibration_curves.csv",
    "energy_by_horizon.csv",
    "updates.csv",
    "configurations.json",
    "model_cards.json",
    "gate_thresholds.json",
    "metadata.json",
    "diagnostic_summary.json",
    "gates.json",
    "perturbation_status.json",
    "example_trajectories.csv",
    "example_bands.csv",
    "example_observed.csv",
    "example_metadata.json",
    "figures/uncal_energy_comparison.png",
    "figures/uncal_coverage_by_horizon.png",
    "figures/uncal_variance_growth.png",
    "figures/uncal_lead_pooling.png",
    "figures/uncal_example_trajectory.png",
    "figures/uncal_pit.png",
    "figure_manifest.json",
    "manifest.json",
]


def _finite(frame: pd.DataFrame, columns: list[str], label: str, failures: list[str]) -> None:
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

    scores = pd.read_csv(OUT / "scores.csv")
    _finite(scores, ["energy_h20", "crps_h20", "coverage_50", "coverage_80", "coverage_95"], "scores", failures)
    if scores[["worm_id", "landmark"]].drop_duplicates().shape[0] != 175:
        failures.append("scores: expected 175 unique landmarks")
    if sorted(scores.worm_id.unique()) != [0, 1, 2, 3, 4]:
        failures.append("scores: expected five sessions")

    neurons = pd.read_csv(OUT / "per_neuron_h20.csv")
    _finite(neurons, ["crps_h20", "coverage_50", "coverage_80", "coverage_95", "width_95"], "per_neuron_h20", failures)
    if not neurons.coverage_95.between(0, 1).all():
        failures.append("per_neuron_h20: coverage outside [0,1]")

    coverage = pd.read_csv(OUT / "coverage_by_horizon.csv")
    _finite(coverage, ["coverage_50", "coverage_80", "coverage_95", "width_95"], "coverage_by_horizon", failures)
    if not coverage[["coverage_50", "coverage_80", "coverage_95"]].stack().between(0, 1).all():
        failures.append("coverage_by_horizon: coverage outside [0,1]")

    variance = pd.read_csv(OUT / "variance_summary.csv")
    if not (variance.within_var_mean >= 0).all():
        failures.append("variance_summary: negative variance")

    repro = json.loads((OUT / "reproduction.json").read_text(encoding="utf-8"))
    if repro.get("gate") != "PASS" or float(repro.get("max_relative_difference", 1)) >= 0.01:
        failures.append("reproduction: Gate A tolerance not met")

    gates = json.loads((OUT / "gates.json").read_text(encoding="utf-8"))
    expected = {
        "A_reproduction": "PASS",
        "D_proper_score_improvement": "PASS",
        "E_marginal_calibration": "FAIL",
        "F_joint_trajectory_fidelity": "FAIL",
        "G_generative_credibility": "FAIL",
        "H_biological_intervention": "NOT_EVALUABLE",
        "stop_label": "PROBABILISTIC CALIBRATION NOT ESTABLISHED",
    }
    for key, value in expected.items():
        if gates.get(key) != value:
            failures.append(f"gates: {key} is {gates.get(key)!r}, expected {value!r}")

    perturbation = json.loads((OUT / "perturbation_status.json").read_text(encoding="utf-8"))
    if perturbation.get("status") != "NOT_LAUNCHED":
        failures.append("perturbation status is not NOT_LAUNCHED")
    if perturbation.get("warning") != "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.":
        failures.append("perturbation warning text is not exact")
    if perturbation.get("biological_validation") != "NOT_EVALUABLE":
        failures.append("biological validation is not NOT_EVALUABLE")

    figures = json.loads((OUT / "figure_manifest.json").read_text(encoding="utf-8"))
    for record in figures:
        path = OUT / "figures" / record["figure"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            failures.append(f"figure manifest mismatch: {record['figure']}")

    ok, problems = verify_manifest(OUT)
    if not ok:
        failures.extend(problems)
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    listed = set(manifest.get("files", {}))
    actual = {str(p.relative_to(OUT)).replace("\\", "/") for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if listed != actual:
        failures.append("manifest: file set does not match report tree")

    if failures:
        print("FAILED")
        print("\n".join(failures))
        return 1
    print(f"validated {len(REQUIRED)} required artifacts; manifest verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
