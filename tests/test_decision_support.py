"""Contract tests for the decision-room support layer."""

from __future__ import annotations

import pytest

from neurobranch.bundlenet_sim.decision_support import (
    UNCONFIRMED_BANNER,
    build_contract,
    contract_markdown,
    option_guidance,
)
from neurobranch.bundlenet_sim.scope import (
    AVAILABLE_DATA,
    PRIMARY_ENDPOINTS,
    PRIMARY_OBJECTS,
    SUCCESS_CRITERIA,
    TARGET_MODES,
    ScopeAnswers,
)
from neurobranch.bundlenet_sim.signed_brief import MATCHES_UNVERIFIED


def complete(**overrides) -> ScopeAnswers:
    values = dict(
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
    values.update(overrides)
    return ScopeAnswers(**values)


@pytest.mark.parametrize("option_id", sorted(PRIMARY_OBJECTS))
def test_every_primary_object_has_source_linked_guidance(option_id: str) -> None:
    guidance = option_guidance("primary_object", option_id)
    assert guidance is not None
    assert guidance.commitment
    assert guidance.requires
    assert guidance.claim_ceiling
    assert guidance.source_note
    assert guidance.label == PRIMARY_OBJECTS[option_id]


@pytest.mark.parametrize("option_id", sorted(PRIMARY_ENDPOINTS))
def test_every_endpoint_has_guidance(option_id: str) -> None:
    assert option_guidance("primary_endpoint", option_id) is not None


@pytest.mark.parametrize("option_id", sorted(AVAILABLE_DATA))
def test_every_data_option_has_guidance(option_id: str) -> None:
    assert option_guidance("available_data", option_id) is not None


@pytest.mark.parametrize("option_id", sorted(SUCCESS_CRITERIA))
def test_every_success_criterion_has_guidance(option_id: str) -> None:
    assert option_guidance("success_criterion", option_id) is not None


@pytest.mark.parametrize("option_id", sorted(TARGET_MODES))
def test_every_target_mode_has_guidance(option_id: str) -> None:
    assert option_guidance("target_mode", option_id) is not None


def test_unknown_or_empty_option_has_no_guidance() -> None:
    assert option_guidance("primary_object", None) is None
    assert option_guidance("primary_object", "invented") is None
    assert option_guidance("unknown", "causal_mechanism") is None


def test_causal_scope_without_intervention_exposes_the_blocker() -> None:
    contract = build_contract(
        complete(available_data=("independent_annotations",), target_mode="not_in_scope")
    )
    assert any("randomized intervention" in item for item in contract.blockers)
    assert "randomized matched intervention" in contract.first_action
    assert "causal" in contract.claim_ceiling.lower()


def test_external_scope_without_sixth_worm_exposes_the_blocker() -> None:
    contract = build_contract(
        complete(
            primary_object="external_generalization",
            available_data=("independent_annotations",),
            target_mode="not_in_scope",
        )
    )
    assert any("sixth held-out worm" in item for item in contract.blockers)
    assert "sixth exact held-out worm" in contract.first_action


def test_bci_scope_is_a_separate_lane() -> None:
    contract = build_contract(complete(primary_object="bci_neuroai"))
    assert any("BCI/NeuroAI lane" in item for item in contract.blockers)
    assert "separate cohort" in contract.first_action
    assert "no transfer" in contract.claim_ceiling or "no transfer" in " ".join(contract.blockers)


def test_contract_rows_are_blank_without_answers() -> None:
    contract = build_contract(ScopeAnswers())
    assert [row.status for row in contract.rows] == ["missing"] * 6
    assert contract.digest == ""
    assert contract.exportable is False
    assert "G0" in contract.first_action


def test_readiness_never_marks_identity_done() -> None:
    contract = build_contract(complete())
    stages = {stage.key: stage for stage in contract.readiness}
    assert stages["g0"].status == "done"
    assert stages["provenance"].status == "done"
    assert stages["identity"].status == "manual"
    assert "No app state" in stages["identity"].detail


def test_digest_stage_requires_the_exported_digest() -> None:
    answers = complete()
    pending = build_contract(answers)
    assert next(s for s in pending.readiness if s.key == "digest").status == "current"
    exported = build_contract(answers, exported_digest=answers.scope_digest())
    assert next(s for s in exported.readiness if s.key == "digest").status == "done"


def test_signed_stage_is_blocked_on_mismatch_and_done_on_unverified_match() -> None:
    answers = complete()
    mismatch = build_contract(answers, signed_status="digest_mismatch")
    assert next(s for s in mismatch.readiness if s.key == "signed").status == "blocked"
    matched = build_contract(answers, signed_status=MATCHES_UNVERIFIED)
    signed = next(s for s in matched.readiness if s.key == "signed")
    assert signed.status == "done"
    assert "identity still manual" in signed.detail
    assert next(s for s in matched.readiness if s.key == "identity").status == "manual"


def test_contract_markdown_is_unconfirmed_and_binds_the_digest() -> None:
    answers = complete()
    contract = build_contract(answers)
    markdown = contract_markdown(contract)
    assert UNCONFIRMED_BANNER in markdown
    assert answers.scope_digest() in markdown
    assert "identity_verification" in markdown
    assert "manual_required" in markdown
    assert "denis confirmed" not in markdown.lower()
    assert "validated" not in markdown.lower()


def test_contract_rendering_does_not_change_the_scope_digest() -> None:
    answers = complete()
    before = answers.scope_digest()
    build_contract(answers)
    contract_markdown(build_contract(answers))
    assert answers.scope_digest() == before
