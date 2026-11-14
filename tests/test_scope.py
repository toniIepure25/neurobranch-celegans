"""Contract tests for the Denis scope-capture model."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from neurobranch.bundlenet_sim.scope import (
    CONFIRMATION_STATUS,
    SCOPE_SCHEMA,
    ScopeAnswers,
)

ROOT = Path(__file__).resolve().parents[1]


def complete_answers() -> ScopeAnswers:
    return ScopeAnswers(
        primary_object="causal_mechanism",
        primary_endpoint="binary_reversal",
        available_data=("randomized_intervention", "independent_annotations"),
        success_criterion="rigorous_negative",
        target_mode="specified",
        target_detail="AVA/AVD ablation with matched sham; optogenetic inhibition.",
        stop_rule="Stop after two independent batches fail the preregistered effect.",
        budget_deadline="6 ECTS, 2027-06-30.",
        recorded_by="Project owner",
        recorded_on="2026-11-20",
        answer_source="email",
    )


def test_empty_scope_has_no_prefilled_answer() -> None:
    answers = ScopeAnswers()
    assert answers.primary_object is None
    assert answers.primary_endpoint is None
    assert answers.available_data == ()
    assert answers.success_criterion is None
    assert answers.target_mode is None
    assert answers.is_complete is False
    assert answers.is_exportable is False
    assert "Primary object is unanswered." in answers.decision_errors()
    assert answers.impact() == (
        "Blocked on G0: primary object unanswered",
        "All new workstreams",
    )


def test_complete_answers_are_exportable_and_map_to_workstream() -> None:
    answers = complete_answers()
    assert answers.is_complete is True
    assert answers.is_exportable is True
    assert answers.impact() == (
        "Randomized intervention + power + independent labels",
        "BCI lane; sixth worm as a goal in itself",
    )
    assert answers.status_label() == "Recorded, not signed"


def test_explicit_not_in_scope_is_a_valid_answer() -> None:
    answers = complete_answers()
    answers.target_mode = "not_in_scope"
    answers.target_detail = ""
    assert answers.is_complete is True
    assert answers.is_exportable is True


def test_other_endpoint_requires_detail() -> None:
    answers = complete_answers()
    answers.primary_endpoint = "other"
    answers.endpoint_detail = " "
    assert "Primary endpoint 'other' needs a written description." in answers.decision_errors()
    answers.endpoint_detail = "Reversal duration"
    assert answers.is_complete is True


def test_none_yet_cannot_be_combined_with_other_data() -> None:
    answers = complete_answers()
    answers.available_data = ("none_yet", "sixth_worm")
    assert "'None yet' cannot be combined with another data item." in answers.decision_errors()


def test_export_requires_provenance() -> None:
    answers = complete_answers()
    answers.recorded_by = ""
    answers.recorded_on = ""
    answers.answer_source = None
    assert answers.is_complete is True
    assert answers.is_exportable is False
    assert "Recorded by is missing." in answers.validate()
    with pytest.raises(ValueError, match="Recorded by is missing"):
        answers.to_payload()


def test_invalid_date_is_rejected() -> None:
    answers = complete_answers()
    answers.recorded_on = "20-11-2026"
    assert "Recorded date must be YYYY-MM-DD." in answers.validate()


def test_payload_round_trip() -> None:
    answers = complete_answers()
    payload = answers.to_payload()
    assert payload["schema"] == SCOPE_SCHEMA
    assert payload["confirmation_status"] == CONFIRMATION_STATUS
    assert payload["workstream"]["first"].startswith("Randomized intervention")
    json.dumps(payload)
    restored = ScopeAnswers.from_payload(payload)
    assert restored == answers


def test_payload_matches_the_published_schema() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "scope_response.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    payload = complete_answers().to_payload()
    jsonschema.validate(instance=payload, schema=schema)

    payload["decisions"]["primary_object"]["id"] = "digital_twin"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=schema)


def test_markdown_is_explicitly_unconfirmed() -> None:
    markdown = complete_answers().to_markdown()
    assert "UNCONFIRMED — NOT SIGNED BY DENIS" in markdown
    assert "confirmation_status" not in markdown
    assert "confirmed by" not in markdown.lower()
    assert "Randomized intervention" in markdown


def test_claim_advisories_are_explicit_and_non_blocking() -> None:
    answers = complete_answers()
    assert answers.is_exportable is True
    codes = [item["code"] for item in answers.claim_advisories()]
    assert codes == ["binary_without_video"]

    answers.available_data = ()
    answers.primary_object = "external_generalization"
    codes = [item["code"] for item in answers.claim_advisories()]
    assert "external_without_sixth_worm" in codes
    assert "external_without_independent_labels" in codes

    answers.primary_object = "causal_mechanism"
    answers.success_criterion = "positive_result"
    codes = [item["code"] for item in answers.claim_advisories()]
    assert "causal_without_intervention" in codes
    assert "causal_without_independent_labels" in codes
    assert "positive_result_without_data" in codes


def test_bci_and_workbench_advisories_do_not_mix_lanes() -> None:
    answers = complete_answers()
    answers.primary_object = "bci_neuroai"
    codes = [item["code"] for item in answers.claim_advisories()]
    assert "bci_without_separate_lane" in codes

    answers.primary_object = "lab_workbench"
    answers.success_criterion = "positive_result"
    codes = [item["code"] for item in answers.claim_advisories()]
    assert "workbench_scientific_success" in codes


def test_advisories_are_in_payload_and_markdown() -> None:
    answers = complete_answers()
    payload = answers.to_payload()
    assert payload["claim_readiness"]["warnings"][0]["code"] == "binary_without_video"
    markdown = answers.to_markdown()
    assert "## Claim-readiness advisories" in markdown
    assert "binary_without_video" in markdown
    assert "not validation errors" in payload["claim_readiness"]["note"]


def test_markdown_has_no_advisory_placeholder_when_none_apply() -> None:
    answers = complete_answers()
    answers.primary_endpoint = "turn_direction"
    markdown = answers.to_markdown()
    assert "- None implied by the recorded answers." in markdown


def test_unknown_values_are_rejected() -> None:
    answers = complete_answers()
    answers.primary_object = "digital_twin"
    answers.primary_endpoint = "all_behaviour"
    answers.success_criterion = "publication"
    answers.answer_source = "rumour"
    errors = answers.validate()
    assert any("Unknown primary object" in item for item in errors)
    assert any("Unknown primary endpoint" in item for item in errors)
    assert any("Unknown success criterion" in item for item in errors)
    assert any("Unknown answer source" in item for item in errors)
