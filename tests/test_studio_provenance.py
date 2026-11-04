"""Unit tests for the version-aware Studio provenance classifier."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "studio_provenance_audit.py"

spec = importlib.util.spec_from_file_location("studio_provenance_audit", SCRIPT)
assert spec and spec.loader
provenance = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = provenance
spec.loader.exec_module(provenance)


def test_line_variants_preserve_content() -> None:
    blob = b"alpha\nbeta\n"
    variants = dict(provenance.line_variants(blob))
    assert variants["git_bytes"] == blob
    assert variants["crlf_bytes"] == b"alpha\r\nbeta\r\n"
    assert provenance.sha256(variants["crlf_bytes"]) == hashlib.sha256(
        b"alpha\r\nbeta\r\n"
    ).hexdigest()


def test_match_kind_distinguishes_exact_and_crlf() -> None:
    blob = b"one\ntwo\n"
    exact = provenance.sha256(blob)
    crlf = b"one\r\ntwo\r\n"
    assert provenance.match_kind(blob, exact, len(blob)) == "git_bytes"
    assert (
        provenance.match_kind(blob, provenance.sha256(crlf), len(crlf))
        == "crlf_bytes"
    )
    assert provenance.match_kind(blob, "0" * 64, len(blob)) is None


def test_dirty_paths_parse_porcelain_and_rename() -> None:
    lines = [
        " M docs/KATO_FINAL_AUDIT_SUPERVISOR_BRIEF.md",
        "?? reports/simulator/kato_final_audit/figures/layout_report.json",
        'R  "old name.py" -> "new name.py"',
    ]
    assert provenance.dirty_paths_from_status(lines) == {
        "docs/KATO_FINAL_AUDIT_SUPERVISOR_BRIEF.md",
        "reports/simulator/kato_final_audit/figures/layout_report.json",
        "new name.py",
    }


def test_current_and_historical_manifests_are_present() -> None:
    assert provenance.STUDIO_MANIFEST.is_file()
    for _, path, _ in provenance.HISTORICAL_MANIFESTS:
        assert path.is_file()
