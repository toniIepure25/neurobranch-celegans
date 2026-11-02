"""Validate the Kato interaction artifacts and guard their claim boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "simulator" / "kato_interactions"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION \u2014 NOT CAUSALLY VALIDATED."

REQUIRED = [
    "analysis_summary.json",
    "baseline_reproduction.csv",
    "reproduction_summary.json",
    "cross_neuron_per_landmark.csv",
    "cross_neuron_summary.csv",
    "cross_neuron_paired.csv",
    "cross_neuron_session.csv",
    "surrogate_draws.csv",
    "surrogate_summary.csv",
    "surrogate_diagnostics.csv",
    "operator_fits.csv",
    "operator_modes.csv",
    "operator_stability.csv",
    "operator_pairwise.csv",
    "local_operator.csv",
    "oscillation_audit.csv",
    "state_transition_scores.csv",
    "state_transition_summary.csv",
    "state_transition_probabilities.csv",
    "named_neuron_presence.csv",
    "selected_sensitivity_landmarks.csv",
    "sensitivity_responses.csv",
    "sensitivity_summary.csv",
    "sensitivity_identifiability.csv",
    "sensitivity_identifiability_summary.csv",
    "synthetic_audit.csv",
    "synthetic_summary.csv",
    "gates.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    checks: dict[str, object] = {}

    missing = [name for name in REQUIRED if not (OUT / name).exists()]
    if missing:
        errors.append(f"missing artifacts: {missing}")

    def read(name: str, **kwargs) -> pd.DataFrame:
        path = OUT / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, **kwargs)

    reproduction = json.loads((OUT / "reproduction_summary.json").read_text(encoding="utf-8"))
    checks["reproduction"] = reproduction
    if reproduction.get("status") != "PASS":
        errors.append("reproduction status is not PASS")
    if reproduction.get("max_abs_reproduction_difference", 1e9) >= 1e-6:
        errors.append("baseline reproduction exceeds 1e-6")
    if abs(reproduction.get("j2_j5_mean_signed_energy_difference", 1e9)) >= 0.01:
        errors.append("J2/J5 mean signed energy difference exceeds 0.01")

    baseline = read("baseline_reproduction.csv")
    if not baseline.empty:
        bad = baseline[baseline.max_abs_diff >= 1e-6]
        if len(bad):
            errors.append(f"{len(bad)} baseline reproduction rows exceed tolerance")
        checks["baseline_metrics"] = sorted(baseline.metric.unique().tolist())

    cross = read("cross_neuron_per_landmark.csv")
    if not cross.empty:
        required_models = {
            "B0_persistence",
            "B1_independent_common",
            "B2_local_mean",
            "B3_population",
            "B4_diagonal_population",
            "B4_refit_diagonal",
            "B5_circular",
            "B5_block",
        }
        absent = sorted(required_models - set(cross.model.unique()))
        if absent:
            errors.append(f"cross-neuron models absent: {absent}")
        counts = cross.groupby("model")["landmark"].size()
        if (counts < 175).any():
            errors.append(f"cross-neuron model has fewer than 175 rows: {counts[counts < 175].to_dict()}")
        if cross[["energy_h20", "trajectory_energy_h20", "mse_mean_h20"]].isna().any().any():
            errors.append("non-finite primary cross-neuron metric")
        checks["cross_models"] = counts.to_dict()

    paired = read("cross_neuron_paired.csv")
    if not paired.empty:
        primary = paired[(paired.block == 5) & (paired.metric == "energy_h20")]
        checks["primary_pairs"] = primary.to_dict(orient="records")
        for reference in ("B1_independent_common", "B4_refit_diagonal", "B5_circular"):
            row = primary[(primary.model == "B3_population") & (primary.reference == reference)]
            if row.empty:
                errors.append(f"missing primary pair B3 vs {reference}")
            elif float(row.iloc[0].ci_low) <= 0:
                errors.append(f"primary pair B3 vs {reference} CI includes zero")

    diag = read("surrogate_diagnostics.csv")
    if not diag.empty:
        med = diag.groupby("method")[["window_acf1_abs_error", "diag_ar_corr"]].median()
        checks["surrogate_diagnostics"] = med.to_dict(orient="index")
        if (med["window_acf1_abs_error"] > 0.10).any():
            errors.append("surrogate lag-1 autocorrelation error exceeds 0.10")
        if (med["diag_ar_corr"] < 0.50).any():
            errors.append("surrogate diagonal AR correlation below 0.50")

    draws = read("surrogate_draws.csv")
    if not draws.empty:
        counts = draws.groupby("method").size()
        checks["surrogate_draw_count"] = counts.to_dict()
        if (counts < 175 * 6).any():
            errors.append(f"surrogate draws below 175*6: {counts.to_dict()}")

    op = read("operator_fits.csv")
    if not op.empty:
        if len(op) != 45:
            errors.append(f"operator_fits has {len(op)} rows, expected 45")
        if (op.spectral_radius > 1.0 + 1e-9).any():
            errors.append("operator spectral radius exceeds one")
        if op[["spectral_radius", "nonnormality", "transient_peak"]].isna().any().any():
            errors.append("non-finite operator metric")
        checks["operator_rows"] = int(len(op))

    modes = read("operator_modes.csv")
    if not modes.empty and not set({"real", "imag", "magnitude"}).issubset(modes.columns):
        errors.append("operator_modes lacks eigenvalue columns")

    probs = read("state_transition_probabilities.csv")
    if not probs.empty:
        if probs.probability.min() < -1e-9 or probs.probability.max() > 1.0 + 1e-9:
            errors.append("state probability outside [0,1]")
        grouped = probs.groupby(["session", "landmark", "model", "from_state"]).probability.sum()
        invalid = grouped[(grouped.sub(1.0).abs() > 1e-6) & (grouped.abs() > 1e-6)]
        if len(invalid):
            errors.append(f"state transition rows do not sum to one: {len(invalid)} rows")
        checks["state_rows"] = int(len(probs))

    presence = read("named_neuron_presence.csv")
    if not presence.empty:
        if len(presence) != 30:
            errors.append(f"named_neuron_presence has {len(presence)} rows, expected 30")
        if set(presence.neuron) != {"AVBL", "AVBR", "AVEL", "AVER", "RIVL", "RIVR"}:
            errors.append("named neuron set changed")
        if (presence.loc[presence.present, "index"] < 0).any():
            errors.append("present named neuron has negative index")
        if (presence.loc[~presence.present, "index"] != -1).any():
            errors.append("absent named neuron was imputed")

    selected = read("selected_sensitivity_landmarks.csv")
    if not selected.empty:
        counts = selected.groupby("session").size()
        if (counts != 3).any():
            errors.append(f"selected sensitivity landmarks are not three per session: {counts.to_dict()}")

    sens = read("sensitivity_responses.csv")
    if not sens.empty:
        if sens.finite_difference_error.max() > 1e-8:
            errors.append("finite-difference disagreement exceeds 1e-8")
        if (sens.ood_fraction < -1e-9).any() or (sens.ood_fraction > 1.0 + 1e-9).any():
            errors.append("OOD fraction outside [0,1]")
        checks["finite_difference_max"] = float(sens.finite_difference_error.max())
        checks["ood_median"] = float(sens.ood_fraction.median())

    summary = json.loads((OUT / "analysis_summary.json").read_text(encoding="utf-8"))
    if summary.get("dataset") != "Kato 2015 WT_NoStim only":
        errors.append("analysis_summary dataset is not the Kato-only dataset")
    if int(summary.get("sessions", 0)) != 5:
        errors.append("analysis_summary sessions is not five")
    if int(summary.get("frames", 0)) != 15662:
        errors.append(f"analysis_summary frames is {summary.get('frames')}, expected 15662")
    checks["analysis_summary"] = summary

    gates = json.loads((OUT / "gates.json").read_text(encoding="utf-8"))
    allowed = {"PASS", "FAIL", "PARTIAL", "INCONCLUSIVE", "NOT_EVALUABLE"}
    if set(gates) != {
        "G1_reproduction",
        "G2_cross_neuron_information",
        "G3_operator_reliability",
        "G4_state_transition_fidelity",
        "G5_sensitivity_reliability",
        "G6_empirical_support",
        "G7_generative_validity",
        "G8_biological_validation",
    }:
        errors.append("gate key set changed")
    for key, gate in gates.items():
        if gate.get("status") not in allowed:
            errors.append(f"{key} has invalid status {gate.get('status')}")
    if gates.get("G8_biological_validation", {}).get("status") != "NOT_EVALUABLE":
        errors.append("biological validation is not NOT_EVALUABLE")
    checks["gates"] = gates

    app = (ROOT / "app" / "handoff_demo.py").read_text(encoding="utf-8")
    if WARNING not in app and 'WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION' not in app:
        errors.append("exact perturbation warning absent from the demo")
    for doc in ("KATO_INTERACTIONS_RESULTS.md", "KATO_INTERACTIONS_LIMITATIONS.md"):
        path = ROOT / "docs" / doc
        if path.exists() and WARNING not in path.read_text(encoding="utf-8"):
            errors.append(f"exact warning absent from {doc}")

    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bad = []
        for entry in manifest.get("files", []):
            path = ROOT / entry["path"]
            if not path.exists() or sha256(path) != entry["sha256"]:
                bad.append(entry["path"])
        checks["manifest_files"] = len(manifest.get("files", []))
        if bad:
            errors.append(f"manifest hash mismatches: {bad[:5]}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": errors}, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
