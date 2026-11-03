"""Validate the frozen Kato final-audit artifacts and manifest hashes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports" / "simulator" / "kato_final_audit"
DOCS = ROOT / "docs"
ALLOWED_TIERS = {
    "SUPPORTED_EXPLORATORY",
    "LIMITED",
    "UNSUPPORTED",
    "INVALIDATED",
    "NOT_EVALUABLE",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_manifest(errors: list[str]) -> None:
    manifest_path = AUDIT / "manifest.json"
    if not manifest_path.exists():
        fail(errors, "missing audit manifest.json")
        return
    manifest = load(manifest_path)
    if manifest.get("dataset", {}).get("sha256", "").lower() != (
        "7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a"
    ):
        fail(errors, "manifest dataset hash mismatch")
    if manifest.get("clean_rerun_status") != "FULL_END_TO_END_REPRODUCTION_PASSED":
        fail(errors, "manifest does not record a passed full reproduction")
    for entry in manifest.get("files", []):
        path = ROOT / entry["path"]
        if not path.exists():
            fail(errors, f"manifest file missing: {entry['path']}")
            continue
        if path.stat().st_size != entry["bytes"]:
            fail(errors, f"manifest size mismatch: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            fail(errors, f"manifest hash mismatch: {entry['path']}")


def main() -> int:
    errors: list[str] = []
    required_docs = [
        "KATO_FINAL_AUDIT_PROTOCOL.md",
        "KATO_FINAL_AUDIT_REPRODUCTION.md",
        "KATO_FINAL_AUDIT_COMPARABILITY.md",
        "KATO_FINAL_AUDIT_SIGN_CONVENTION.md",
        "KATO_FINAL_AUDIT_UNCERTAINTY.md",
        "KATO_FINAL_AUDIT_SCIENTIFIC_CLAIMS.md",
        "KATO_FINAL_AUDIT_RESULTS.md",
        "KATO_FINAL_AUDIT_SUPERVISOR_BRIEF.md",
    ]
    for name in required_docs:
        if not (DOCS / name).exists():
            fail(errors, f"missing doc: {name}")

    required_artifacts = [
        "preflight.json",
        "full_rerun_status.json",
        "reproduction_comparison.json",
        "uncertainty_bootstrap.csv",
        "uncertainty_summary.json",
        "session_estimates.csv",
        "leave_one_session_out.csv",
        "paired_serial_dependence.csv",
        "landmark_subsample_sensitivity.csv",
        "horizon_uncertainty.csv",
        "ten_factor_operators.csv",
        "ten_factor_sessions.csv",
        "ten_factor_summary.json",
        "model_comparability.csv",
        "parameter_counts.csv",
        "decomposition_order_sensitivity.csv",
        "comparability_summary.json",
        "sign_convention_decision.json",
        "claims_ledger.json",
        "decision_record.json",
        "manifest.json",
    ]
    for name in required_artifacts:
        if not (AUDIT / name).exists():
            fail(errors, f"missing artifact: {name}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    reproduction = load(AUDIT / "reproduction_comparison.json")
    if reproduction.get("status") != "FULL_END_TO_END_REPRODUCTION_PASSED":
        fail(errors, "clean rerun status is not PASSED")
    if reproduction.get("missing_files"):
        fail(errors, "comparison reports missing files")
    if not reproduction.get("all_numeric_within_1e-8"):
        fail(errors, "comparison exceeds 1e-8")
    for comparison in reproduction.get("comparisons", []):
        if comparison.get("max_abs_difference", 0) > 1e-8:
            fail(errors, f"difference above 1e-8: {comparison['file']}")
    gate_comparison = next(
        (item for item in reproduction["comparisons"] if item["file"] == "gates.json"),
        None,
    )
    if not gate_comparison or gate_comparison.get("status") != "GATE_STATUS_MATCH":
        fail(errors, "gate status mismatch")
    gates = gate_comparison.get("gate_statuses", {}) if gate_comparison else {}
    expected = {
        "G1_historical_reproduction": "PASS",
        "G2_fair_matched_baselines": "PASS",
        "G3_common_factor_explanation": "PASS",
        "G4_residual_cross_neuron_information": "PASS",
        "G5_synthetic_specificity": "PASS",
        "G6_robustness": "PASS",
        "G7_structural_identifiability": "FAIL",
        "G8_biological_intervention_validity": "NOT_EVALUABLE",
    }
    for gate, status in expected.items():
        if gates.get(gate) != status:
            fail(errors, f"gate {gate} status {gates.get(gate)!r} != {status!r}")

    sign = load(AUDIT / "sign_convention_decision.json")
    if sign.get("decision") != "B_GENUINE_RULE_AMBIGUITY_NOT_FULLY_PRESPECIFIED":
        fail(errors, "sign decision was not downgraded")
    if sign.get("g4_confirmatory_status") != "NOT_ESTABLISHED":
        fail(errors, "G4 confirmatory status is not NOT_ESTABLISHED")

    ledger = load(AUDIT / "claims_ledger.json")
    tiers = {claim["evidence_tier"] for claim in ledger.get("claims", [])}
    if not tiers <= ALLOWED_TIERS:
        fail(errors, f"invalid evidence tier(s): {sorted(tiers - ALLOWED_TIERS)}")
    if "CONFIRMED" in tiers:
        fail(errors, "CONFIRMED tier is forbidden")
    if ledger.get("no_confirmed_label") is not True:
        fail(errors, "ledger does not declare no CONFIRMED label")
    claims = {claim["claim_id"]: claim for claim in ledger.get("claims", [])}
    if claims.get("C05", {}).get("evidence_tier") != "SUPPORTED_EXPLORATORY":
        fail(errors, "primary residual claim is not SUPPORTED_EXPLORATORY")
    if claims.get("C07", {}).get("evidence_tier") != "INVALIDATED":
        fail(errors, "51/49 uniqueness claim is not INVALIDATED")
    if claims.get("C22", {}).get("evidence_tier") != "INVALIDATED":
        fail(errors, "confirmatory G4 claim is not INVALIDATED")
    if any("CONFIRMED" in json.dumps(c.get("evidence_tier", "")) for c in claims.values()):
        fail(errors, "a claim uses CONFIRMED")

    decision = load(AUDIT / "decision_record.json")
    if decision.get("decision") != "READY_FOR_SUPERVISOR_REVIEW":
        fail(errors, "decision is not READY_FOR_SUPERVISOR_REVIEW")
    if decision.get("reproduction_status") != "FULL_END_TO_END_REPRODUCTION_PASSED":
        fail(errors, "decision reproduction status mismatch")
    if decision.get("g4_confirmatory") != "NOT_ESTABLISHED":
        fail(errors, "decision G4 caveat missing")

    import pandas as pd

    blocks = pd.read_csv(AUDIT / "uncertainty_bootstrap.csv")
    if set(blocks["block"]) != {3, 5, 10, 20}:
        fail(errors, "bootstrap blocks are not 3/5/10/20")
    if (blocks["ci_low"] <= 0).any():
        fail(errors, "a bootstrap block interval crosses zero")
    sessions = pd.read_csv(AUDIT / "session_estimates.csv")
    if int((sessions["mean_advantage"] > 0).sum()) != 4:
        fail(errors, "expected four positive session estimates")
    if not (sessions["mean_advantage"] < 0).any():
        fail(errors, "expected a reversing session")
    ten = load(AUDIT / "ten_factor_summary.json")
    if ten.get("evaluation_pca_is_common_factor_pca") is not True:
        fail(errors, "evaluation PCA is not the common-factor PCA")
    ratio = ten.get("b3_residual_unexplained_ratio", {})
    if not (0.9 <= ratio.get("median", 0) < 1.0):
        fail(errors, "ten-factor residual diagnostic is not in the expected range")
    comp = load(AUDIT / "comparability_summary.json")
    shares = comp.get("order_sensitivity", [])
    if len(shares) != 3:
        fail(errors, "decomposition does not contain three baselines")
    for entry in shares:
        if abs(entry["common_share"] + entry["residual_share"] - 1.0) > 1e-12:
            fail(errors, f"shares do not sum to 1: {entry['baseline']}")
    common_shares = {round(entry["common_share"], 3) for entry in shares}
    if len(common_shares) < 3:
        fail(errors, "decomposition shares are not baseline-dependent")

    figures = [
        "kato_final_audit_uncertainty.png",
        "kato_final_audit_decomposition.png",
        "kato_final_audit_summary.png",
    ]
    for name in figures:
        path = AUDIT / "figures" / name
        if not path.exists():
            fail(errors, f"missing figure: {name}")
            continue
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            fail(errors, f"not a PNG: {name}")
        if path.stat().st_size < 10_000:
            fail(errors, f"figure is implausibly small: {name}")

    validate_manifest(errors)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("KATO FINAL AUDIT VALIDATION PASSED")
    print(f"reproduction: {reproduction['status']}")
    print(f"claims: {ledger['counts']}")
    print(f"decision: {decision['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
