"""Tests for the Kato final-audit evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports" / "simulator" / "kato_final_audit"


def load(name: str) -> dict:
    return json.loads((AUDIT / name).read_text(encoding="utf-8"))


def test_eight_audit_docs_exist() -> None:
    names = [
        "KATO_FINAL_AUDIT_PROTOCOL.md",
        "KATO_FINAL_AUDIT_REPRODUCTION.md",
        "KATO_FINAL_AUDIT_COMPARABILITY.md",
        "KATO_FINAL_AUDIT_SIGN_CONVENTION.md",
        "KATO_FINAL_AUDIT_UNCERTAINTY.md",
        "KATO_FINAL_AUDIT_SCIENTIFIC_CLAIMS.md",
        "KATO_FINAL_AUDIT_RESULTS.md",
        "KATO_FINAL_AUDIT_SUPERVISOR_BRIEF.md",
    ]
    for name in names:
        assert (ROOT / "docs" / name).exists(), name


def test_clean_rerun_passed_with_exact_comparison() -> None:
    result = load("reproduction_comparison.json")
    assert result["status"] == "FULL_END_TO_END_REPRODUCTION_PASSED"
    assert result["missing_files"] == []
    assert result["all_numeric_within_1e-8"] is True
    assert all(item.get("max_abs_difference", 0) <= 1e-8 for item in result["comparisons"])
    assert result["key_metrics"]["B3_advantage_B5_minus_B3"]["abs_difference"] == 0.0
    gates = next(item for item in result["comparisons"] if item["file"] == "gates.json")
    assert gates["status"] == "GATE_STATUS_MATCH"
    assert gates["gate_statuses"]["G7_structural_identifiability"] == "FAIL"
    assert gates["gate_statuses"]["G8_biological_intervention_validity"] == "NOT_EVALUABLE"


def test_rerun_completion_was_clean() -> None:
    status = load("full_rerun_status.json")
    assert status["completed_normally"] is True
    assert status["stderr_bytes"] == 0
    assert status["reported_elapsed_seconds"] > 0
    assert status["artifact_files"] >= 20


def test_bootstrap_blocks_all_positive() -> None:
    table = pd.read_csv(AUDIT / "uncertainty_bootstrap.csv")
    assert set(table["block"]) == {3, 5, 10, 20}
    assert (table["ci_low"] > 0).all()
    assert (table["ci_high"] > 0).all()
    assert (table["mean_advantage"] == table["mean_advantage"].iloc[0]).all()


def test_sessions_are_heterogeneous_not_independent_animals() -> None:
    table = pd.read_csv(AUDIT / "session_estimates.csv")
    assert int((table["mean_advantage"] > 0).sum()) == 4
    assert float(table["mean_advantage"].min()) < 0
    assert len(table) == 5


def test_horizon_reversal_is_preserved() -> None:
    table = pd.read_csv(AUDIT / "horizon_uncertainty.csv").set_index("horizon")
    assert table.loc[1, "mean_advantage"] < 0
    assert table.loc[1, "ci_high"] < 0
    for horizon in (5, 10, 20):
        assert table.loc[horizon, "mean_advantage"] > 0
        assert table.loc[horizon, "ci_low"] > 0


def test_decomposition_is_ordered_not_unique() -> None:
    summary = load("comparability_summary.json")
    rows = summary["order_sensitivity"]
    assert len(rows) == 3
    for row in rows:
        assert abs(row["common_share"] + row["residual_share"] - 1.0) < 1e-12
    assert len({round(row["common_share"], 3) for row in rows}) == 3
    assert abs(summary["reported_decomposition"]["common_B4m_minus_B5"] - 0.2750793961657263) < 1e-12
    assert abs(summary["reported_decomposition"]["residual_B5_minus_B3"] - 0.264437992668356) < 1e-12


def test_ten_factor_control_uses_same_pca_and_has_residual_restriction() -> None:
    summary = load("ten_factor_summary.json")
    assert summary["evaluation_pca_is_common_factor_pca"] is True
    assert summary["pca_fit"].startswith("training frames only")
    ratio = summary["b3_residual_unexplained_ratio"]
    assert 0.98 <= ratio["min"] < ratio["median"] < ratio["max"] < 1.0
    operators = pd.read_csv(AUDIT / "ten_factor_operators.csv")
    assert operators["session"].nunique() == 5
    assert len(operators) >= 175
    assert "b3_residual_unexplained_ratio_median" in operators.columns


def test_sign_is_downgraded_without_changing_numbers() -> None:
    sign = load("sign_convention_decision.json")
    assert sign["decision"] == "B_GENUINE_RULE_AMBIGUITY_NOT_FULLY_PRESPECIFIED"
    assert sign["g4_numerical_status"] == "PASS_UNDER_CORRECTED_SIGN"
    assert sign["g4_confirmatory_status"] == "NOT_ESTABLISHED"
    assert sign["claim_label"] == "SUPPORTED_EXPLORATORY"
    assert sign["numerical_result_retained"]["positive_b3_advantage"] == 0.264437992668356


def test_claims_ledger_has_no_confirmed_tier() -> None:
    ledger = load("claims_ledger.json")
    allowed = {"SUPPORTED_EXPLORATORY", "LIMITED", "UNSUPPORTED", "INVALIDATED", "NOT_EVALUABLE"}
    assert {claim["evidence_tier"] for claim in ledger["claims"]} <= allowed
    assert ledger["no_confirmed_label"] is True
    assert sum(ledger["counts"].values()) == len(ledger["claims"])
    by_id = {claim["claim_id"]: claim for claim in ledger["claims"]}
    assert by_id["C05"]["evidence_tier"] == "SUPPORTED_EXPLORATORY"
    assert by_id["C07"]["evidence_tier"] == "INVALIDATED"
    assert by_id["C19"]["evidence_tier"] == "NOT_EVALUABLE"
    assert by_id["C22"]["evidence_tier"] == "INVALIDATED"


def test_decision_record_is_ready_but_not_confirmatory() -> None:
    record = load("decision_record.json")
    assert record["decision"] == "READY_FOR_SUPERVISOR_REVIEW"
    assert record["reproduction_status"] == "FULL_END_TO_END_REPRODUCTION_PASSED"
    assert record["g4_confirmatory"] == "NOT_ESTABLISHED"
    assert record["structural_identifiability"] == "FAIL"
    assert record["biological_intervention_validity"] == "NOT_EVALUABLE"


def test_figures_are_real_pngs() -> None:
    for name in (
        "kato_final_audit_uncertainty.png",
        "kato_final_audit_decomposition.png",
        "kato_final_audit_summary.png",
    ):
        path = AUDIT / "figures" / name
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert path.stat().st_size > 10_000
