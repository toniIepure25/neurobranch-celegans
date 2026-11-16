"""Tests for the portable gap-closure handoff packet."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from neurobranch.bundlenet_sim.handoff import (
    PACKET_FILES,
    build_decision_packet,
    build_gap_packet,
)
from neurobranch.bundlenet_sim.scope import ScopeAnswers

ROOT = Path(__file__).resolve().parents[1]


def _complete_answers() -> ScopeAnswers:
    return ScopeAnswers(
        primary_object="causal_mechanism",
        primary_endpoint="binary_reversal",
        available_data=("independent_annotations",),
        success_criterion="rigorous_negative",
        target_mode="not_in_scope",
        stop_rule="Stop after two independent batches fail.",
        budget_deadline="6 ECTS, 2027-06-30.",
        recorded_by="QA transcriber",
        recorded_on="2026-11-20",
        answer_source="email",
    )


def test_packet_contains_documents_and_hashes() -> None:
    data = build_gap_packet(ROOT)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        assert "MANIFEST.json" in names
        assert "gap-closure/docs/DENIS_REQUIREMENTS_BRIEF.md" in names
        assert "gap-closure/docs/DENIS_PUBLIC_SOURCE_AUDIT.md" in names
        assert "gap-closure/schemas/scope_response.schema.json" in names
        assert "gap-closure/schemas/signed_brief_audit.schema.json" in names
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


def test_decision_packet_contains_contract_response_and_manifest() -> None:
    answers = _complete_answers()
    assert answers.is_exportable
    payload = answers.to_payload()
    digest = answers.scope_digest()
    data = build_decision_packet(
        ROOT,
        contract_markdown="# Contract\n",
        answers_markdown="# Answers\n",
        payload=payload,
        scope_digest=digest,
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        assert "MANIFEST.json" in names
        assert "decision-packet/docs/DENIS_REQUIREMENTS_BRIEF.md" in names
        assert "decision-packet/docs/DENIS_PUBLIC_SOURCE_AUDIT.md" in names
        assert "decision-packet/docs/DATA_REQUEST_PACKAGE.md" in names
        assert "decision-packet/docs/GAP_CLOSURE_MATRIX.md" in names
        assert "decision-packet/schemas/scope_response.schema.json" in names
        assert "decision-packet/schemas/signed_brief_audit.schema.json" in names
        assert "decision-packet/denis_scope_contract.md" in names
        assert "decision-packet/denis_scope_response.json" in names
        assert "decision-packet/denis_scope_response.md" in names
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["format"] == "vienna-decision-packet-v1"
        assert manifest["confirmation_status"] == "unconfirmed_transcription"
        assert manifest["scope_digest"] == digest
        assert "not a signature" in manifest["caution"]
        assert len(manifest["files"]) == 9
        for entry in manifest["files"]:
            assert len(entry["sha256"]) == 64
            body = archive.read(entry["path"])
            assert len(body) == entry["bytes"]
            assert hashlib.sha256(body).hexdigest() == entry["sha256"]
        assert archive.read("decision-packet/denis_scope_contract.md") == b"# Contract\n"


def test_decision_packet_is_deterministic() -> None:
    answers = _complete_answers()
    kwargs = {
        "contract_markdown": "# Contract\n",
        "answers_markdown": "# Answers\n",
        "payload": answers.to_payload(),
        "scope_digest": answers.scope_digest(),
    }
    assert build_decision_packet(ROOT, **kwargs) == build_decision_packet(ROOT, **kwargs)


def test_decision_packet_rejects_a_digest_mismatch() -> None:
    answers = _complete_answers()
    with pytest.raises(ValueError, match="does not match the payload"):
        build_decision_packet(
            ROOT,
            contract_markdown="# Contract\n",
            answers_markdown="# Answers\n",
            payload=answers.to_payload(),
            scope_digest="a" * 64,
        )


def test_decision_packet_fails_loudly_when_a_document_is_missing(tmp_path: Path) -> None:
    answers = _complete_answers()
    with pytest.raises(FileNotFoundError, match="DENIS_REQUIREMENTS_BRIEF"):
        build_decision_packet(
            tmp_path,
            contract_markdown="# Contract\n",
            answers_markdown="# Answers\n",
            payload=answers.to_payload(),
            scope_digest=answers.scope_digest(),
        )
