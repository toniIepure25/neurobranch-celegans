"""Schema and consistency checks for generated Phase 9 reports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE9 = ROOT / "reports" / "phase9"


@pytest.mark.skipif(not PHASE9.exists(), reason="Phase 9 reports not generated")
def test_reports_validate_against_schemas() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_reports.py"), "--strict"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not (PHASE9 / "claims.json").exists(), reason="claims ledger not generated")
def test_claim_statuses_are_from_the_fixed_vocabulary() -> None:
    payload = json.loads((PHASE9 / "claims.json").read_text(encoding="utf-8"))
    allowed = {"supported", "not_supported", "bounded_null", "underpowered"}
    assert all(claim["status"] in allowed for claim in payload["claims"])
    assert all(claim["evidence"] for claim in payload["claims"])


@pytest.mark.skipif(not (PHASE9 / "claims.json").exists(), reason="claims ledger not generated")
def test_underpowered_claims_report_an_mde() -> None:
    payload = json.loads((PHASE9 / "claims.json").read_text(encoding="utf-8"))
    for claim in payload["claims"]:
        if claim["status"] == "underpowered":
            assert claim["mde"] is not None
            assert claim["margin"] is not None
            assert claim["mde"] > claim["margin"]
