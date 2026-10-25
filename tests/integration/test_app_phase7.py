"""Integration checks for Phase 7 documentation, configuration and app tab."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION"


def test_phase7_documents_and_config_exist() -> None:
    names = [
        "PHASE7_RESEARCH_SYNTHESIS.md",
        "PHASE7_PREREGISTRATION.md",
        "PHASE7_DEVIATIONS.md",
        "PHASE7_RESULTS.md",
        "PHASE7_SUPERVISOR_BRIEF.md",
        "PHASE7_EVALUATION_CARD.md",
    ]
    for name in names:
        path = ROOT / "docs" / name
        assert path.is_file(), name
        assert WARNING in path.read_text(encoding="utf-8"), name
    assert (ROOT / "configs" / "phase7" / "active.yaml").is_file()


def test_phase7_artifacts_when_present_are_auditable() -> None:
    root = ROOT / "reports" / "phase7"
    metrics = root / "metrics.csv"
    audit = root / "audit.json"
    if not metrics.exists() or not audit.exists():
        pytest.skip("Phase 7 full run has not been generated in this checkout")
    import json

    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["no_test_target_selected"] is True
    assert payload["fold_leaks"] == []
    assert metrics.stat().st_size > 0
