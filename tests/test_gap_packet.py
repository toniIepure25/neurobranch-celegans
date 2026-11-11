"""Tests for the portable gap-closure handoff packet."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from neurobranch.bundlenet_sim.handoff import PACKET_FILES, build_gap_packet

ROOT = Path(__file__).resolve().parents[1]


def test_packet_contains_documents_and_hashes() -> None:
    data = build_gap_packet(ROOT)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        assert "MANIFEST.json" in names
        assert "gap-closure/docs/DENIS_REQUIREMENTS_BRIEF.md" in names
        assert "gap-closure/reports/vienna_workbench/intervention_power.json" in names
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["format"] == "vienna-gap-closure-packet-v1"
        assert len(manifest["files"]) == len(PACKET_FILES)
        for entry in manifest["files"]:
            assert len(entry["sha256"]) == 64
            assert entry["bytes"] >= 0
            assert archive.read(f"gap-closure/{entry['path']}")


def test_packet_is_deterministic() -> None:
    assert build_gap_packet(ROOT) == build_gap_packet(ROOT)


def test_packet_fails_loudly_when_a_document_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="DENIS_REQUIREMENTS_BRIEF"):
        build_gap_packet(tmp_path)
