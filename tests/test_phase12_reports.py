"""Contract tests for the generated Phase 12 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE12 = ROOT / "reports" / "phase12"


def test_summary_matches_schema() -> None:
    schema = json.loads(
        (ROOT / "src" / "neurobranch" / "schemas" / "phase12_summary.schema.json").read_text()
    )
    summary = json.loads((PHASE12 / "summary.json").read_text())
    jsonschema.validate(summary, schema)


def test_gate_status_is_fail_closed() -> None:
    summary = json.loads((PHASE12 / "summary.json").read_text())
    gates = summary["gates"]
    all_passed = all(
        gates[name] for name in ("g1_sbc", "g2_heldout_ppc", "g3_negative_control", "g4_prior_sensitivity")
    )
    assert summary["status"] == ("adopted" if all_passed else "not_adopted_gate_failures")
    assert summary["hierarchical"]["pooled_ceiling_median"] == pytest.approx(
        summary["hierarchical"]["pooled_reliability_median"] ** 0.5, abs=1e-9
    )


def test_hierarchical_target_table_contract() -> None:
    frame = pd.read_csv(PHASE12 / "hierarchical_by_target.csv")
    assert len(frame) == 259
    ok = frame[frame["status"] == "ok"]
    assert len(ok) == 126
    for column in ("reliability_n1", "reliability_n22", "ceiling_n22", "grid_edge_mass"):
        assert column in frame.columns
    assert ok["reliability_n22"].between(0, 1).all()
    assert (ok["reliability_ci_low_n22"] <= ok["reliability_n22"]).all()
    assert (ok["reliability_n22"] <= ok["reliability_ci_high_n22"]).all()


def test_benchmark_has_no_supported_claim() -> None:
    summary = json.loads((PHASE12 / "benchmark_cluster.json").read_text())
    frame = pd.read_csv(PHASE12 / "benchmark_cluster.csv")
    assert summary["n_supported"] == 0
    assert not frame["status"].eq("supported").any()
    assert (frame["mde_80"] > summary["sesoi"]).all()


def test_manifest_lists_every_artifact() -> None:
    manifest = json.loads((PHASE12 / "manifest.json").read_text())
    expected = {
        path.name
        for path in PHASE12.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }
    assert set(manifest["artifacts"]) == expected
    assert manifest["phase"] == "phase12"
