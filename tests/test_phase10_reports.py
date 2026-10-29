"""Schema and consistency checks for generated Phase 10 reports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE10 = ROOT / "reports" / "phase10"


@pytest.mark.skipif(not PHASE10.exists(), reason="Phase 10 reports not generated")
def test_reports_validate_against_schemas() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_reports.py"), "--strict"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not (PHASE10 / "claims.json").exists(), reason="Phase 10 claims ledger not generated")
def test_claim_statuses_are_from_the_fixed_vocabulary() -> None:
    payload = json.loads((PHASE10 / "claims.json").read_text(encoding="utf-8"))
    allowed = {"supported", "not_supported", "bounded_null", "underpowered"}
    assert all(claim["status"] in allowed for claim in payload["claims"])
    assert all(claim["evidence"] for claim in payload["claims"])


@pytest.mark.skipif(not (PHASE10 / "claims.json").exists(), reason="Phase 10 claims ledger not generated")
def test_underpowered_claims_report_an_mde() -> None:
    payload = json.loads((PHASE10 / "claims.json").read_text(encoding="utf-8"))
    for claim in payload["claims"]:
        if claim["status"] == "underpowered":
            assert claim["mde"] is not None
            assert claim["margin"] is not None
            assert claim["mde"] > claim["margin"]


@pytest.mark.skipif(not (PHASE10 / "reliability_summary.json").exists(), reason="Phase 10 summary not generated")
def test_phase10_metric_aligned_ceilings_are_separated() -> None:
    summary = json.loads((PHASE10 / "reliability_summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 2
    assert summary["fold_size"] >= 1
    assert 0.0 <= summary["pooled_reliability_at_fold"] <= 1.0
    assert summary["pooled_ceiling_at_fold"] > summary["mean_animal_ceiling"]
    assert summary["n_pooled_targets"] == summary["n_animal_targets"]


@pytest.mark.skipif(not (PHASE10 / "claims.json").exists(), reason="Phase 10 claims ledger not generated")
def test_phase10_claims_carry_estimands() -> None:
    payload = json.loads((PHASE10 / "claims.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    folded = [claim for claim in payload["claims"] if claim["id"] == "phase10-fold-ceiling"]
    assert folded and folded[0]["estimand"] == "pooled_fold"
    assert folded[0]["n_eval_animals"] == 22
