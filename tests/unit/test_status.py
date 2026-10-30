"""Tests for the dashboard measurement-status loader."""

from __future__ import annotations

import json
from pathlib import Path

from neurobranch.reporting.status import load_measurement_status


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prefers_phase14_and_reports_missing_gate_as_failed(tmp_path: Path) -> None:
    _write(
        tmp_path / "phase14" / "summary.json",
        {
            "decision": "not_adopted_gate_failures",
            "model": "m4_spike_slab",
            "real_fit": {"pooled_reliability_median_n22": 0.01},
            "reference": {"v1_pooled_reliability_at_fold": 0.3005},
            "gates": {"G1a": {"passed": False, "value": 0.92}},
        },
    )
    status = load_measurement_status(tmp_path)
    assert status is not None
    assert status.phase == "phase14"
    assert status.adopted is False
    assert status.failed_gates == ["G1a"]
    assert status.primary_estimate == 0.01


def test_falls_back_to_phase13_then_labels_phase12_historical(tmp_path: Path) -> None:
    _write(
        tmp_path / "phase13" / "summary.json",
        {
            "decision": "not_adopted_gate_failures",
            "model": "m2_event_variance",
            "primary_estimate": 0.2,
            "reference_estimate": 0.5600069018705827,
            "gates": {"G1": {"passed": False}},
        },
    )
    status = load_measurement_status(tmp_path)
    assert status is not None and status.phase == "phase13"

    (tmp_path / "phase13" / "summary.json").unlink()
    _write(
        tmp_path / "phase12" / "summary.json",
        {"status": "not_adopted", "hierarchical": {"pooled_reliability_median": 0.1}},
    )
    fallback = load_measurement_status(tmp_path)
    assert fallback is not None
    assert fallback.phase == "phase12"
    assert fallback.historical is True


def test_missing_reports_returns_none(tmp_path: Path) -> None:
    assert load_measurement_status(tmp_path) is None
