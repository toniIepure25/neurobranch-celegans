"""Contract tests for the signed-brief consistency audit."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from neurobranch.bundlenet_sim.scope import ScopeAnswers
from neurobranch.bundlenet_sim.signed_brief import (
    BLOCKED_ON_TRANSCRIPTION,
    DIGEST_MISMATCH,
    INCOMPLETE,
    INVALID_FORMAT,
    MATCHES_UNVERIFIED,
    NO_DIGEST,
    SIGNED_BRIEF_AUDIT_SCHEMA,
    SignedBrief,
)

ROOT = Path(__file__).resolve().parents[1]


def complete_answers() -> ScopeAnswers:
    return ScopeAnswers(
        primary_object="causal_mechanism",
        primary_endpoint="binary_reversal",
        available_data=("randomized_intervention", "independent_annotations"),
        success_criterion="rigorous_negative",
        target_mode="specified",
        target_detail="AVA/AVD ablation with matched sham.",
        stop_rule="Stop after two independent batches fail.",
        budget_deadline="6 ECTS, 2027-06-30.",
        recorded_by="Project owner",
        recorded_on="2026-11-20",
        answer_source="email",
    )


def block(
    digest: str | None = None,
    *,
    confirmed_by: str = "QA reviewer",
    date: str = "2026-11-20",
    primary_object: str = "A causal test of a named-neuron or circuit perturbation",
) -> str:
    digest_value = complete_answers().scope_digest() if digest is None else digest
    return "\n".join(
        [
            "```text",
            f"Primary object: {primary_object}",
            "Primary endpoint: Binary reversal onset/offset",
            "Available data: randomized intervention; independent annotation",
            "Success criterion: A rigorous negative result",
            "Target / intervention (if any): AVA/AVD ablation with matched sham",
            "Stop rule: Stop after two independent batches fail.",
            "Budget / deadline: 6 ECTS, 2027-06-30.",
            f"Scope digest: {digest_value}",
            f"Confirmed by: {confirmed_by}  Date: {date}",
            "```",
        ]
    )


def test_complete_block_matches_without_identity_claim() -> None:
    brief = SignedBrief.parse_markdown(block())
    assert brief.missing_fields() == ()
    assert brief.format_errors == ()
    comparison = brief.compare(complete_answers())
    assert comparison.status == MATCHES_UNVERIFIED
    assert comparison.digest_match is True
    assert comparison.signed_digest == complete_answers().scope_digest()
    assert "identity is not verified" in comparison.note
    assert brief.confirmed_by == "QA reviewer"
    assert brief.date == "2026-11-20"


def test_uppercase_digest_and_backticks_are_normalised() -> None:
    digest = complete_answers().scope_digest().upper()
    brief = SignedBrief.parse_markdown(block(f"`{digest}`"))
    assert brief.scope_digest == digest.lower()
    assert brief.compare(complete_answers()).status == MATCHES_UNVERIFIED


def test_date_may_be_on_a_separate_line() -> None:
    text = block().replace(
        "Confirmed by: QA reviewer  Date: 2026-11-20",
        "Confirmed by: QA reviewer\nDate: 2026-11-20",
    )
    brief = SignedBrief.parse_markdown(text)
    assert brief.confirmed_by == "QA reviewer"
    assert brief.date == "2026-11-20"
    assert brief.compare(complete_answers()).status == MATCHES_UNVERIFIED


def test_missing_placeholder_fields_are_reported() -> None:
    text = block(primary_object="").replace(
        "Stop rule: Stop after two independent batches fail.", "Stop rule: ______"
    )
    brief = SignedBrief.parse_markdown(text)
    missing = brief.missing_fields()
    assert "Primary object" in missing
    assert "Stop rule" in missing
    assert brief.compare(complete_answers()).status == INCOMPLETE


def test_no_fenced_block_is_invalid() -> None:
    brief = SignedBrief.parse_markdown("Primary object: no fence")
    assert brief.format_errors
    assert brief.compare(complete_answers()).status == INVALID_FORMAT


def test_duplicate_fields_are_invalid() -> None:
    text = block().replace(
        "Stop rule: Stop after two independent batches fail.",
        "Stop rule: one\nStop rule: two",
    )
    brief = SignedBrief.parse_markdown(text)
    assert "Duplicate field: Stop rule." in brief.format_errors
    assert brief.compare(complete_answers()).status == INVALID_FORMAT


def test_invalid_date_is_incomplete() -> None:
    brief = SignedBrief.parse_markdown(block(date="20-11-2026"))
    comparison = brief.compare(complete_answers())
    assert comparison.status == INCOMPLETE
    assert "YYYY-MM-DD" in comparison.note


def test_malformed_digest_is_incomplete() -> None:
    brief = SignedBrief.parse_markdown(block("not-a-digest"))
    comparison = brief.compare(complete_answers())
    assert comparison.status == INCOMPLETE
    assert "malformed" in comparison.note


def test_no_digest_is_legacy_not_binding() -> None:
    text = block().replace(
        f"Scope digest: {complete_answers().scope_digest()}",
        "Scope digest: ______",
    )
    brief = SignedBrief.parse_markdown(text)
    comparison = brief.compare(complete_answers())
    assert comparison.status == NO_DIGEST
    assert comparison.digest_match is False


def test_edited_transcription_produces_digest_mismatch() -> None:
    answers = complete_answers()
    brief = SignedBrief.parse_markdown(block())
    answers.stop_rule = "A different stop rule."
    comparison = brief.compare(answers)
    assert comparison.status == DIGEST_MISMATCH
    assert comparison.signed_digest != comparison.current_digest


def test_incomplete_transcription_blocks_comparison() -> None:
    brief = SignedBrief.parse_markdown(block())
    comparison = brief.compare(ScopeAnswers())
    assert comparison.status == BLOCKED_ON_TRANSCRIPTION
    assert comparison.current_digest == ""


def test_audit_payload_matches_schema_and_markdown_keeps_caveat() -> None:
    brief = SignedBrief.parse_markdown(block())
    comparison = brief.compare(complete_answers())
    payload = brief.to_audit_payload(comparison)
    assert payload["schema"] == SIGNED_BRIEF_AUDIT_SCHEMA
    assert payload["identity_verification"] == "manual_required"
    schema = json.loads(
        (ROOT / "schemas" / "signed_brief_audit.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=payload, schema=schema)
    markdown = brief.to_audit_markdown(comparison).lower()
    assert "identity not verified" in markdown
    assert "not an authentication result" in markdown
    assert "denis confirmed" not in markdown


def test_audit_schema_rejects_a_verified_claim() -> None:
    brief = SignedBrief.parse_markdown(block())
    payload = brief.to_audit_payload(brief.compare(complete_answers()))
    payload["identity_verification"] = "verified"
    schema = json.loads(
        (ROOT / "schemas" / "signed_brief_audit.schema.json").read_text(encoding="utf-8")
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=schema)
