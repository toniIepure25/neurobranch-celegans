"""Decision support for the Vienna scope room.

This module turns the six recorded answers into a source-linked contract. It
does not infer answers, authenticate a signer, or promote a scope into evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neurobranch.bundlenet_sim.scope import (
    AVAILABLE_DATA,
    PRIMARY_ENDPOINTS,
    PRIMARY_OBJECTS,
    SUCCESS_CRITERIA,
    TARGET_MODES,
    ScopeAnswers,
)
from neurobranch.bundlenet_sim.signed_brief import (
    DIGEST_MISMATCH,
    INCOMPLETE,
    INVALID_FORMAT,
    MATCHES_UNVERIFIED,
    NO_DIGEST,
)

UNCONFIRMED_BANNER = "UNCONFIRMED — NOT SIGNED BY DENIS"

DECISION_LABELS: dict[str, str] = {
    "primary_object": "Primary object",
    "primary_endpoint": "Primary endpoint",
    "available_data": "Available data",
    "success_criterion": "Success criterion",
    "target_mode": "Target / intervention",
    "stop_rule": "Stop rule and budget",
}

_OBJECT_GUIDANCE: dict[str, dict[str, Any]] = {
    "causal_mechanism": {
        "commitment": "Design or test a named-neuron or circuit perturbation.",
        "requires": (
            "Randomized intervention with matched sham.",
            "Independent behaviour labels.",
            "Replication before a causal effect is claimed.",
        ),
        "claim_ceiling": (
            "An association or an intervention design until the randomized, replicated "
            "contrast exists; never a causal proof from the current assets."
        ),
        "source_note": "Group project framing; G3 in the gap register.",
    },
    "external_generalization": {
        "commitment": "Test whether the representation or forecast survives a new animal.",
        "requires": (
            "A sixth held-out worm with the exact BunDLe-Net embedding, or an external cohort.",
            "Independent labels for the chosen endpoint.",
        ),
        "claim_ceiling": (
            "Within-release held-out evidence on five worms; no population claim and no "
            "sixth-worm generalization until that animal exists."
        ),
        "source_note": "BunDLe-Net release (five worms); G1 in the gap register.",
    },
    "lab_workbench": {
        "commitment": "Ship a reusable intake, QC, annotation and provenance workbench.",
        "requires": (
            "A frozen intake schema and acceptance gates.",
            "A benchmark the group can rerun.",
        ),
        "claim_ceiling": (
            "Product and protocol readiness; no new biological or causal result."
        ),
        "source_note": "The workbench's own intake and provenance contracts.",
    },
    "bci_neuroai": {
        "commitment": "Scope a separate BCI/NeuroAI translation lane.",
        "requires": (
            "A separate cohort and endpoint.",
            "Ethics approval and a BIDS/QC path.",
            "Subject-level splits and a leakage audit.",
        ),
        "claim_ceiling": (
            "A separate feasibility claim; no transfer from C. elegans and no mixing with "
            "the worm results."
        ),
        "source_note": "G4 in the gap register; no BCI/EEG/fMRI data are in this release.",
    },
}

_ENDPOINT_GUIDANCE: dict[str, dict[str, Any]] = {
    "eight_states": {
        "commitment": "Use the released eight behavioural states as the endpoint.",
        "requires": ("Independent eight-behaviour video annotation for a behavioural claim.",),
        "claim_ceiling": "Agreement with released labels; not independent behavioural validation.",
        "source_note": "NC-MCM release; G2 in the gap register.",
    },
    "binary_reversal": {
        "commitment": "Use reversal onset/offset as a binary endpoint.",
        "requires": (
            "Raw traces plus video for an independent onset/offset definition.",
            "An explicit interval and offset convention.",
        ),
        "claim_ceiling": (
            "A retrospective association in the current audit; the prospective test does not "
            "support forecasting a reversal."
        ),
        "source_note": "Tier-3 external reversal audit; G2 in the gap register.",
    },
    "turn_direction": {
        "commitment": "Use dorsal versus ventral turn direction.",
        "requires": ("A labelled turn-direction dataset and a frozen definition.",),
        "claim_ceiling": "Not implemented; no claim until the endpoint and labels exist.",
        "source_note": "Gap register; no turn-direction endpoint is shipped.",
    },
    "continuous_kinematics": {
        "commitment": "Use a continuous kinematic variable.",
        "requires": ("A defined kinematic variable, video and preprocessing.",),
        "claim_ceiling": "Not implemented; no claim until the variable is defined and measured.",
        "source_note": "Gap register; no continuous-kinematics endpoint is shipped.",
    },
    "other": {
        "commitment": "Define a different endpoint in writing.",
        "requires": ("A written definition and a matching labelled dataset.",),
        "claim_ceiling": "No claim until the endpoint is specified and measured.",
        "source_note": "G0 detail field; no inferred endpoint.",
    },
}

_DATA_GUIDANCE: dict[str, dict[str, Any]] = {
    "sixth_worm": {
        "commitment": "A sixth held-out worm with the exact BunDLe-Net consistent embedding.",
        "requires": ("No overlap with the five release worms; exact pipeline and commit.",),
        "claim_ceiling": "One-animal external check; not a population claim.",
        "source_note": "BunDLe-Net release; G1.",
    },
    "raw_traces_video": {
        "commitment": "Raw traces plus behaviour video for independent onset/offset labels.",
        "requires": ("A frozen annotation protocol and blinded annotators.",),
        "claim_ceiling": "Independent behavioural agreement, if the labels pass the gate.",
        "source_note": "G2 in the gap register.",
    },
    "randomized_intervention": {
        "commitment": "A randomized optogenetic/ablation intervention with matched sham.",
        "requires": ("Randomization, blinding, a common clock and a pre-registered contrast.",),
        "claim_ceiling": "A causal effect in that assay only after replication.",
        "source_note": "G3 in the gap register.",
    },
    "independent_annotations": {
        "commitment": "An independent eight-behaviour video annotation.",
        "requires": ("Two blinded annotators and a pre-registered agreement threshold.",),
        "claim_ceiling": "Independent label agreement; not a causal result by itself.",
        "source_note": "G2 in the gap register.",
    },
    "none_yet": {
        "commitment": "No named data asset is available yet.",
        "requires": ("Obtain one of the named assets before running the current models.",),
        "claim_ceiling": "No new scientific claim.",
        "source_note": "G0 data answer.",
    },
}

_SUCCESS_GUIDANCE: dict[str, dict[str, Any]] = {
    "positive_result": {
        "commitment": "Success means a positive predictive or causal result.",
        "requires": ("A pre-registered decision rule and data that can test it.",),
        "claim_ceiling": "Only the result the design identifies; a positive result is not automatic.",
        "source_note": "G0 success criterion.",
    },
    "rigorous_negative": {
        "commitment": "Success includes a rigorous negative result.",
        "requires": ("A pre-registered negative-control and stopping rule.",),
        "claim_ceiling": "A bounded negative finding, not evidence for a mechanism.",
        "source_note": "G0 success criterion.",
    },
    "reusable_benchmark": {
        "commitment": "Success means a reusable benchmark or protocol.",
        "requires": ("Frozen inputs, baselines, uncertainty and a rerun path.",),
        "claim_ceiling": "Benchmark readiness; no biological claim from the benchmark alone.",
        "source_note": "G0 success criterion.",
    },
    "wet_lab_design": {
        "commitment": "Success means a design for a wet-lab experiment.",
        "requires": ("A protocol, power calculation and ethical/feasibility review.",),
        "claim_ceiling": "A design deliverable; not an executed experiment.",
        "source_note": "G0 success criterion.",
    },
    "working_demo": {
        "commitment": "Success means a working product demo for the group.",
        "requires": ("A stable demo path and explicit non-claims.",),
        "claim_ceiling": "Product demo only; not scientific validation.",
        "source_note": "G0 success criterion.",
    },
}

_TARGET_GUIDANCE: dict[str, dict[str, Any]] = {
    "not_in_scope": {
        "commitment": "No intervention is in scope for this project.",
        "requires": (),
        "claim_ceiling": "No intervention claim; causal language remains out of scope.",
        "source_note": "Explicit target answer.",
    },
    "specified": {
        "commitment": "A named circuit, modality and positive control will be specified.",
        "requires": (
            "A written target and positive control.",
            "Randomized intervention data and a matched sham.",
        ),
        "claim_ceiling": "A design target until the randomized contrast exists.",
        "source_note": "G3 in the gap register.",
    },
}


def option_guidance(decision: str, option_id: str | None) -> OptionGuidance | None:
    """Return source-linked guidance for one selected option, or None."""
    if not option_id:
        return None
    if decision == "primary_object":
        table, label = _OBJECT_GUIDANCE, PRIMARY_OBJECTS
    elif decision == "primary_endpoint":
        table, label = _ENDPOINT_GUIDANCE, PRIMARY_ENDPOINTS
    elif decision == "available_data":
        table, label = _DATA_GUIDANCE, AVAILABLE_DATA
    elif decision == "success_criterion":
        table, label = _SUCCESS_GUIDANCE, SUCCESS_CRITERIA
    elif decision == "target_mode":
        table, label = _TARGET_GUIDANCE, TARGET_MODES
    else:
        return None
    item = table.get(option_id)
    if item is None or option_id not in label:
        return None
    return OptionGuidance(
        decision=decision,
        option_id=option_id,
        label=label[option_id],
        commitment=str(item["commitment"]),
        requires=tuple(item["requires"]),
        claim_ceiling=str(item["claim_ceiling"]),
        source_note=str(item["source_note"]),
    )


def _row(label: str, value: str, status: str = "answered") -> ContractRow:
    return ContractRow(label=label, value=value or "Unanswered", status=status if value else "missing")


def _claim_ceiling(answers: ScopeAnswers) -> str:
    if answers.primary_object in _OBJECT_GUIDANCE:
        return str(_OBJECT_GUIDANCE[answers.primary_object]["claim_ceiling"])
    return "No claim until the primary object is answered; G0 is still open."


def _first_action(answers: ScopeAnswers) -> str:
    if not answers.is_complete:
        return "G0 is open: ask Denis the six questions and record his written answer; do not collect new data."
    advisories = {item["code"] for item in answers.claim_advisories()}
    if "causal_without_intervention" in advisories or "intervention_without_randomization" in advisories:
        return "Design or obtain a randomized matched intervention and independent labels."
    if "external_without_sixth_worm" in advisories:
        return "Acquire the sixth exact held-out worm, or reframe the claim as within-release."
    if answers.primary_object == "lab_workbench":
        return "Ship the intake, QC, annotation and provenance workbench with a rerunnable benchmark."
    if answers.primary_object == "bci_neuroai":
        return "Scope the separate cohort, ethics, BIDS/QC and subject-level split before modelling."
    if "none_yet_scope" in advisories:
        return "Obtain one named data asset before running the current models."
    return "Export the transcription, return it for signature, then check the signer identity manually."


def _readiness(
    answers: ScopeAnswers,
    *,
    exported_digest: str,
    signed_status: str,
) -> tuple[ReadinessStage, ...]:
    digest = answers.scope_digest() if answers.is_exportable else ""
    if not answers.is_complete:
        g0 = ReadinessStage("g0", "Six decisions", "current", "G0 is open; nothing is inferred.")
        provenance = ReadinessStage("provenance", "Provenance", "pending", "Answer all six decisions first.")
        digest_stage = ReadinessStage("digest", "Digest exported", "pending", "No export yet.")
        signed_stage = ReadinessStage("signed", "Signed block consistent", "pending", "No signed block checked.")
    elif not answers.is_exportable:
        g0 = ReadinessStage("g0", "Six decisions", "done", "All six decisions are recorded.")
        provenance = ReadinessStage("provenance", "Provenance", "current", "Record who transcribed what and when.")
        digest_stage = ReadinessStage("digest", "Digest exported", "pending", "No export yet.")
        signed_stage = ReadinessStage("signed", "Signed block consistent", "pending", "No signed block checked.")
    else:
        g0 = ReadinessStage("g0", "Six decisions", "done", "All six decisions are recorded.")
        provenance = ReadinessStage("provenance", "Provenance", "done", "Provenance is complete.")
        if digest and exported_digest == digest:
            digest_stage = ReadinessStage("digest", "Digest exported", "done", "Export digest recorded in this session.")
        else:
            digest_stage = ReadinessStage("digest", "Digest exported", "current", "Download the JSON or contract to record the digest.")
        if signed_status == MATCHES_UNVERIFIED:
            signed_stage = ReadinessStage("signed", "Signed block consistent", "done", "Fields and digest match; identity still manual.")
        elif signed_status in {DIGEST_MISMATCH, INCOMPLETE, INVALID_FORMAT, NO_DIGEST}:
            signed_stage = ReadinessStage("signed", "Signed block consistent", "blocked", "The signed block does not bind to this transcription.")
        else:
            signed_stage = ReadinessStage("signed", "Signed block consistent", "pending", "No signed block checked.")

    identity = ReadinessStage(
        "identity",
        "Signer identity",
        "manual",
        "No app state can close this gate; a human must check the signer.",
    )
    claim = ReadinessStage("claim", "Claim ceiling", "current", _claim_ceiling(answers))
    return (g0, provenance, digest_stage, signed_stage, identity, claim)


def build_contract(
    answers: ScopeAnswers,
    *,
    exported_digest: str = "",
    signed_status: str = "",
) -> ScopeContract:
    """Build a live contract from answers without changing their meaning."""
    data_labels = " · ".join(AVAILABLE_DATA[item] for item in answers.available_data)
    endpoint_value = PRIMARY_ENDPOINTS.get(answers.primary_endpoint or "", "")
    if answers.primary_endpoint == "other" and answers.endpoint_detail:
        endpoint_value = f"{endpoint_value}: {answers.endpoint_detail}"
    target_value = TARGET_MODES.get(answers.target_mode or "", "")
    if answers.target_mode == "specified" and answers.target_detail:
        target_value = f"{target_value}: {answers.target_detail}"
    stop_value = " · ".join(
        value for value in (answers.stop_rule.strip(), answers.budget_deadline.strip()) if value
    )

    rows = (
        _row("Primary object", PRIMARY_OBJECTS.get(answers.primary_object or "", "")),
        _row("Primary endpoint", endpoint_value),
        _row("Available data", data_labels),
        _row("Success criterion", SUCCESS_CRITERIA.get(answers.success_criterion or "", "")),
        _row("Target / intervention", target_value),
        _row("Stop rule and budget", stop_value),
    )
    blockers: list[str] = []
    if not answers.is_complete:
        blockers.extend(answers.decision_errors())
    blockers.extend(item["text"] for item in answers.claim_advisories())
    return ScopeContract(
        rows=rows,
        claim_ceiling=_claim_ceiling(answers),
        blockers=tuple(dict.fromkeys(blockers)),
        first_action=_first_action(answers),
        readiness=_readiness(answers, exported_digest=exported_digest, signed_status=signed_status),
        digest=answers.scope_digest() if answers.is_exportable else "",
        exportable=answers.is_exportable,
    )


def contract_markdown(contract: ScopeContract) -> str:
    """Render the contract with an explicit unconfirmed banner."""
    blocker_lines = [f"- {blocker}" for blocker in contract.blockers] or ["- None recorded."]
    lines = [
        "# Scope decision contract",
        "",
        f"> **{UNCONFIRMED_BANNER}.** This is a working contract derived from a transcription;",
        "> it is not a signed requirement and it cannot authenticate the signer.",
        "",
        "| Decision | Answer | Status |",
        "|---|---|---|",
    ]
    lines.extend(f"| {row.label} | {row.value} | {row.status} |" for row in contract.rows)
    lines.extend(
        [
            "",
            f"**Claim ceiling:** {contract.claim_ceiling}",
            "",
            f"**First action:** {contract.first_action}",
            "",
            "## Readiness",
            "",
            "| Stage | State | Detail |",
            "|---|---|---|",
        ]
    )
    lines.extend(f"| {stage.label} | {stage.status} | {stage.detail} |" for stage in contract.readiness)
    lines.extend(
        [
            "",
            "## Blockers",
            "",
            *blocker_lines,
            "",
            f"**Scope digest:** `{contract.digest or '—'}`",
            "",
            "**identity_verification:** `manual_required`",        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class OptionGuidance:
    decision: str
    option_id: str
    label: str
    commitment: str
    requires: tuple[str, ...]
    claim_ceiling: str
    source_note: str


@dataclass(frozen=True)
class ContractRow:
    label: str
    value: str
    status: str


@dataclass(frozen=True)
class ReadinessStage:
    key: str
    label: str
    status: str
    detail: str


@dataclass(frozen=True)
class ScopeContract:
    rows: tuple[ContractRow, ...]
    claim_ceiling: str
    blockers: tuple[str, ...]
    first_action: str
    readiness: tuple[ReadinessStage, ...]
    digest: str
    exportable: bool
