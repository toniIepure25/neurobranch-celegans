"""Integration checks for Phase 8 documentation, config and dashboard tab."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION"


def test_phase8_documents_and_config_exist() -> None:
    names = [
        "PHASE8_PREREGISTRATION.md",
        "PHASE8_RESEARCH_SYNTHESIS.md",
        "PHASE8_APT.md",
        "PHASE8_EVALUATION_CARD.md",
        "PHASE8_DEVIATIONS.md",
        "PHASE8_RESULTS.md",
    ]
    for name in names:
        path = ROOT / "docs" / name
        assert path.is_file(), name
        assert WARNING in path.read_text(encoding="utf-8"), name
    assert (ROOT / "configs" / "phase8" / "apt.yaml").is_file()


def test_phase8_app_tab_exists() -> None:
    text = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Adaptive Twin" in text
    assert "Phase 8" in text
    assert "MODEL-DEPENDENT VIRTUAL PERTURBATION" in text


def test_phase8_artifacts_when_present_are_auditable() -> None:
    root = ROOT / "reports" / "phase8"
    summary = root / "summary.json"
    audit = root / "audit.json"
    readout = root / "readout.json"
    if not summary.exists() or not audit.exists():
        pytest.skip("Phase 8 full run has not been generated in this checkout")
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["animal_overlaps"] == []
    assert payload["support_test_target_overlaps"] == []
    assert payload["support_test_event_overlaps"] == []
    assert payload["test_event_in_train_or_val"] == []
    assert (root / "metrics.csv").stat().st_size > 0
    if readout.exists():
        r = json.loads(readout.read_text(encoding="utf-8"))
        assert r["warning"].startswith("MODEL-DEPENDENT VIRTUAL PERTURBATION")
        assert "H1_primary" in r["decisions"]
