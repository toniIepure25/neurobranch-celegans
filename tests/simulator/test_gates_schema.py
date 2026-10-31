"""Gate report schema and six-gate contract."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "simulator" / "gates.json"
SCHEMA = ROOT / "schemas" / "simulator_gates.schema.json"
PACKAGED = ROOT / "src" / "neurobranch" / "schemas" / "simulator_gates.schema.json"


def test_gate_report_matches_schema() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    assert set(payload["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert payload["gates"]["G6"]["status"] == "not_evaluable"


def test_schema_copies_match() -> None:
    assert SCHEMA.read_text(encoding="utf-8") == PACKAGED.read_text(encoding="utf-8")
