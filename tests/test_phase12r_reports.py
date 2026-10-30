"""Contract tests for the Phase 12R correction artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE12R = ROOT / "reports" / "phase12r"
pytestmark = pytest.mark.skipif(
    not (PHASE12R / "summary.json").exists(), reason="Phase 12R artifacts not generated"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summary_matches_schema() -> None:
    schema = json.loads(
        (ROOT / "src" / "neurobranch" / "schemas" / "phase12r_summary.schema.json").read_text()
    )
    summary = json.loads((PHASE12R / "summary.json").read_text())
    jsonschema.validate(summary, schema)


def test_gate_status_is_fail_closed() -> None:
    summary = json.loads((PHASE12R / "summary.json").read_text())
    gates = summary["gates"]
    components = (
        gates["g1a_sbc_coverage"],
        gates["g1b_sbc_rank"],
        gates["g1c_grid_edge"],
        gates["g1d_fit_integrity"],
    )
    assert gates["g1_sbc_all"] == all(components)
    all_passed = gates["g1_sbc_all"] and all(
        gates[name] for name in ("g2_heldout_ppc", "g3_negative_control", "g4_prior_sensitivity")
    )
    assert gates["all_gates"] == all_passed
    if summary["mode"] == "quick":
        assert summary["status"] == "quick_mode_no_decision"
    elif all_passed and summary["estimate_within_0.05"]:
        assert summary["status"] == "adopted"
    elif all_passed:
        assert summary["status"] == "not_adopted_estimate_mismatch"
    else:
        assert summary["status"] == "not_adopted_gate_failures"


def test_hierarchical_target_table_contract() -> None:
    frame = pd.read_csv(PHASE12R / "hierarchical_by_target.csv")
    assert len(frame) == 259
    assert (frame["status"] == "ok").sum() == 126
    for column in ("reason", "reliability_n1", "reliability_n22", "ceiling_n22", "grid_edge_mass"):
        assert column in frame.columns
    ok = frame[frame["status"] == "ok"]
    assert ok["reliability_n22"].between(0, 1).all()
    assert (ok["reliability_ci_low_n22"] <= ok["reliability_n22"]).all()
    assert (ok["reliability_n22"] <= ok["reliability_ci_high_n22"]).all()
    assert frame.loc[frame["status"] != "ok", "reason"].notna().all()


def test_primary_estimate_and_sensitivity_are_reported() -> None:
    summary = json.loads((PHASE12R / "summary.json").read_text())
    table = pd.read_csv(PHASE12R / "min_events_sensitivity.csv")
    assert set(table["min_events"]) == {1, 2}
    primary = table.loc[table["min_events"] == 1].iloc[0]
    assert summary["hierarchical"]["pooled_reliability_median"] == pytest.approx(
        primary["pooled_reliability_median"], abs=1e-12
    )
    assert summary["hierarchical"]["pooled_ceiling_median"] == pytest.approx(
        summary["hierarchical"]["pooled_reliability_median"] ** 0.5, abs=1e-9
    )


def test_manifest_lists_every_artifact() -> None:
    manifest = json.loads((PHASE12R / "manifest.json").read_text())
    expected = {
        path.name
        for path in PHASE12R.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }
    assert set(manifest["artifacts"]) == expected
    assert manifest["phase"] == "phase12r"
    for name, digest in manifest["artifacts"].items():
        assert _sha256(PHASE12R / name) == digest
