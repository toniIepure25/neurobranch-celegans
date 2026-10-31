"""Tests for the Phase 6 threshold audit's pure calculations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "phase6_threshold_audit", ROOT / "scripts/phase6_threshold_audit.py"
)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_support_audit_reports_q95_rollout_drop_and_ratio():
    rows = []
    for family, worm, t90, t95, t99, r90, r95, r99 in [
        ("var", 0, 10.0, 11.0, 12.0, 1.0, 1.0, 1.0),
        ("switching_var", 1, 9.0, 10.0, 11.0, 0.70, 0.80, 0.99),
    ]:
        rows.extend(
            [
                {
                    "family": family,
                    "worm_id": worm,
                    "quantile": q,
                    "threshold": threshold,
                    "rollout_in_support": rollout,
                    "observed_test_in_support": 0.95,
                }
                for q, threshold, rollout in ((0.90, t90, r90), (0.95, t95, r95), (0.99, t99, r99))
            ]
        )
    result = AUDIT._support_audit(pd.DataFrame(rows))
    assert result["q95_over_q99_min"] == 10.0 / 11.0
    assert result["q95_rollout_min"] == 0.80
    assert result["n_family_worm_below_90pct_at_q95"] == 1
    assert result["n_family_worm_below_90pct_at_q99"] == 0


def test_lf_hash_normalizes_windows_line_endings(tmp_path):
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{"a": 1}\n{"b": 2}\n')
    crlf.write_bytes(b'{"a": 1}\r\n{"b": 2}\r\n')
    assert AUDIT._lf_sha256(lf) == AUDIT._lf_sha256(crlf)
