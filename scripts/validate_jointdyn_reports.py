"""Validate the joint-dynamics report contract, frozen results and manifest."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neurobranch.adaptive.manifest import sha256_file, verify_manifest
from neurobranch.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "reports/simulator/joint_dynamics"
DOCS = PROJECT_ROOT / "docs"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."
REQUIRED = [
    "autocorr_profile.csv",
    "bootstrap_block_sensitivity.csv",
    "calibration_curves.csv",
    "calibration_h20.csv",
    "calibration_models.csv",
    "configurations.json",
    "covariance_h20.csv",
    "coverage_by_horizon.csv",
    "coverage_by_landmark.csv",
    "coverage_summary.csv",
    "d4_decision.json",
    "d4_neuron_h20.csv",
    "d4_scores_by_landmark.csv",
    "d4_summary.csv",
    "d4_validation_grid.csv",
    "d4_validation_summary.csv",
    "energy_by_horizon.csv",
    "example_bands.csv",
    "example_metadata.json",
    "example_observed.csv",
    "example_trajectories.csv",
    "factorial_decomposition.csv",
    "figure_manifest.json",
    "gates.json",
    "gate_thresholds.json",
    "historical_block_scores.csv",
    "innovation_acf.csv",
    "innovation_decision.json",
    "innovation_factors.csv",
    "innovation_reconstruction.csv",
    "innovation_structure.csv",
    "model_cards.json",
    "neural_subset_scores.csv",
    "neuron_h20.csv",
    "paired_scores.csv",
    "parameter_decision.json",
    "parameter_draws.csv",
    "parameter_scores.csv",
    "parameter_summary.csv",
    "parameter_synthetic_audit.csv",
    "reproduction_check.csv",
    "scores_by_landmark.csv",
    "score_summary.csv",
    "secondary_paired.csv",
    "sensitivity_representation.csv",
    "session_paired_ci.csv",
    "session_paired_scores.csv",
    "session_scores.csv",
    "trajectory_examples.npz",
    "uncertainty_decomposition.csv",
    "stage_status.csv",
    "negative_findings.csv",
    "perturbation_status.json",
    "metadata.json",
    "manifest.json",
    "figures/jd_autocorrelation.png",
    "figures/jd_calibration.png",
    "figures/jd_covariance.png",
    "figures/jd_example_trajectory.png",
    "figures/jd_factorial.png",
    "figures/jd_parameter.png",
    "figures/jd_pit.png",
    "figures/jd_scores.png",
    "figures/jd_trajectories.html",
]
EXPECTED = {
    "J0_persistence": (2.950334, 2.285787, 0.452406, 0.493781, 0.790664, 0.936477),
    "J1_independent_diag": (3.145713, 2.435121, 0.418209, 0.471117, 0.745612, 0.889816),
    "J2_population_diag": (2.700283, 2.057628, 0.413679, 0.586474, 0.845666, 0.944428),
    "J3_population_full": (3.010946, 2.276567, 0.459414, 0.766965, 0.956984, 0.991801),
    "J4_independent_full": (3.007244, 2.358549, 0.424165, 0.559871, 0.819506, 0.934450),
    "J5_population_lead1_boot": (2.700150, 2.059962, 0.413370, 0.587781, 0.848665, 0.946998),
}


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

    scores = pd.read_csv(OUT / "scores_by_landmark.csv")
    _finite(
        scores,
        [
            "energy_h20",
            "trajectory_energy_h20",
            "crps_h20",
            "crps_h20_full",
            "coverage_h20_50",
            "coverage_h20_80",
            "coverage_h20_95",
            "coverage_h20_full_50",
            "coverage_h20_full_80",
            "coverage_h20_full_95",
            "neural_cov_error",
            "autocorr_error",
        ],
        "scores_by_landmark",
        failures,
    )
    if scores.groupby(["worm_id", "landmark"]).ngroups != 175:
        failures.append("scores_by_landmark: expected 175 unique landmarks")
    if sorted(scores.worm_id.unique().tolist()) != [0, 1, 2, 3, 4]:
        failures.append("scores_by_landmark: expected five sessions")
    if set(scores.model.unique()) != set(EXPECTED):
        failures.append("scores_by_landmark: model set changed")

    summary = pd.read_csv(OUT / "score_summary.csv").set_index("model")
    columns = ["energy_h20", "trajectory_energy_h20", "crps_h20_full", "coverage_h20_full_50", "coverage_h20_full_80", "coverage_h20_full_95"]
    for model, values in EXPECTED.items():
        if model not in summary.index:
            failures.append(f"score_summary: missing {model}")
            continue
        for column, expected in zip(columns, values):
            if not np.isclose(float(summary.loc[model, column]), expected, atol=2e-4):
                failures.append(f"score_summary: {model} {column} changed")

    paired = pd.read_csv(OUT / "paired_scores.csv")
    row = paired[(paired.model == "J5_population_lead1_boot") & (paired.reference == "J1_independent_diag") & (paired.metric == "energy_h20")]
    if row.empty or not (float(row.mean_improvement.iloc[0]) > 0 and float(row.ci_low.iloc[0]) > 0):
        failures.append("paired_scores: G2 contrast not positive")

    repro = pd.read_csv(OUT / "reproduction_check.csv")
    if float(repro.relative_difference.max()) >= 0.02:
        failures.append("reproduction: G1 tolerance not met")

    innovation = json.loads((OUT / "innovation_decision.json").read_text(encoding="utf-8"))
    if innovation.get("temporal_model_justified") is not False:
        failures.append("innovation_decision: temporal model unexpectedly justified")
    pop = innovation.get("median_by_model", {}).get("population_var", {})
    if float(pop.get("acf1", 1)) > 0.15 or float(pop.get("acf5", 1)) > 0.10:
        failures.append("innovation_decision: ACF gate not met")

    gates = json.loads((OUT / "gates.json").read_text(encoding="utf-8"))
    expected_gates = {
        "G1_reproduction": "PASS",
        "G2_population_information": "PASS",
        "G3_innovation_validity": "PASS",
        "G4_marginal_calibration": "FAIL",
        "G5_joint_trajectory": "FAIL",
        "G6_robustness": "PASS",
        "G7_generative_credibility": "FAIL",
        "G8_biological_intervention": "NOT_EVALUABLE",
    }
    for key, value in expected_gates.items():
        if gates.get(key, {}).get("status") != value:
            failures.append(f"gates: {key} is not {value}")
    if gates.get("stop_label") != "PROBABILISTIC CALIBRATION NOT ESTABLISHED":
        failures.append("gates: stop label changed")

    d4 = json.loads((OUT / "d4_decision.json").read_text(encoding="utf-8"))
    if d4.get("accepted") is not False:
        failures.append("d4_decision: D4 unexpectedly accepted")
    parameter = json.loads((OUT / "parameter_decision.json").read_text(encoding="utf-8"))
    if parameter.get("accepted") is not False:
        failures.append("parameter_decision: parameter ensemble unexpectedly accepted")

    perturbation = json.loads((OUT / "perturbation_status.json").read_text(encoding="utf-8"))
    if perturbation.get("status") != "NOT_LAUNCHED":
        failures.append("perturbation: status is not NOT_LAUNCHED")
    if perturbation.get("warning") != WARNING:
        failures.append("perturbation: warning text is not exact")
    if perturbation.get("biological_validation") != "NOT_EVALUABLE":
        failures.append("perturbation: biological validation is not NOT_EVALUABLE")

    for name in [
        "JOINTDYN_PROTOCOL.md",
        "JOINTDYN_REPRODUCTION.md",
        "JOINTDYN_POPULATION_INFORMATION.md",
        "JOINTDYN_INNOVATION_STRUCTURE.md",
        "JOINTDYN_CALIBRATION.md",
        "JOINTDYN_PARAMETER_UNCERTAINTY.md",
        "JOINTDYN_TRAJECTORY_FIDELITY.md",
        "JOINTDYN_RESULTS.md",
        "JOINTDYN_LIMITATIONS.md",
        "JOINTDYN_SUPERVISOR_BRIEF.md",
        "JOINTDYN_NEXT_EXPERIMENTS.md",
    ]:
        if not (DOCS / name).exists():
            failures.append(f"docs: missing {name}")
    for name in ["JOINTDYN_PROTOCOL.md", "JOINTDYN_RESULTS.md", "JOINTDYN_LIMITATIONS.md", "JOINTDYN_SUPERVISOR_BRIEF.md"]:
        if WARNING not in (DOCS / name).read_text(encoding="utf-8"):
            failures.append(f"docs: exact warning missing in {name}")

    figures = json.loads((OUT / "figure_manifest.json").read_text(encoding="utf-8"))
    for record in figures:
        path = OUT / "figures" / record["figure"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            failures.append(f"figure manifest mismatch: {record['figure']}")

    payload = np.load(OUT / "trajectory_examples.npz")
    if len(payload.files) != 1225:
        failures.append(f"trajectory_examples: expected 1225 arrays, found {len(payload.files)}")
    if any(name.endswith((".mat", ".pt", ".pth")) for name in payload.files):
        failures.append("trajectory_examples: raw recording or weights embedded")

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
    print(f"validated {len(REQUIRED)} required artifacts; gates, docs, figures and manifest verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
