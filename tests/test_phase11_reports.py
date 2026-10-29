"""Contract tests for the Phase 11 artifacts and their stated caveats."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE11 = ROOT / "reports" / "phase11"


def _json(name: str) -> dict:
    return json.loads((PHASE11 / name).read_text(encoding="utf-8"))


def test_coverage_sensitivity_is_monotone_and_matches_headline() -> None:
    table = pd.read_csv(PHASE11 / "coverage_sensitivity.csv")
    assert table["pooled_reliability_at_fold"].is_monotonic_decreasing
    row = table[table["min_responder_coverage"] == 0.8].iloc[0]
    headline = _json("summary.json")["headline"]
    assert row["pooled_reliability_at_fold"] == headline["coverage_0_8_pooled_reliability"]
    assert row["pooled_ceiling_at_fold"] == math.sqrt(row["pooled_reliability_at_fold"])


def test_benchmark_card_uses_coverage_aware_ceiling() -> None:
    card = _json("benchmark_card.json")
    assert abs(card["ceiling"]["value"] - 0.5600069018705827) < 1e-9
    assert card["claim_statuses"]["phase10-fold-ceiling"] == "not_supported"


def test_failed_gates_are_recorded_not_hidden() -> None:
    extrapolation = _json("extrapolation.json")
    assert not extrapolation["gate_passed"]
    shrinkage = _json("half_replication.json")
    assert shrinkage["shrinkage_gate_passed"] is False
    uncertainty = _json("uncertainty_summary.json")
    assert uncertainty["interval_status"] == "failed_point_outside_interval"
    assert uncertainty["point_estimate_in_interval"] is False
    synthetic = _json("coverage_study.json")
    assert synthetic["coverage"] >= 0.97
    assert synthetic["mean_width"] < 0.30
