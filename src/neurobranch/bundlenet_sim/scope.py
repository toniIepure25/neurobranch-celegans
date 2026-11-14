"""Capture Denis's six scope decisions without inventing an answer.

The public record does not contain a Denis-specific brief. This module provides a
small, testable representation of the six decisions in
``docs/DENIS_REQUIREMENTS_BRIEF.md`` and an export that is explicitly marked as
an unconfirmed transcription. A completed form is not a signature.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

SCOPE_SCHEMA = "vienna-scope-response-v1"
CONFIRMATION_STATUS = "unconfirmed_transcription"

PRIMARY_OBJECTS: dict[str, str] = {
    "causal_mechanism": "A causal test of a named-neuron or circuit perturbation",
    "external_generalization": "External generalization of the representation/forecast",
    "lab_workbench": "A reusable lab workbench (intake, QC, annotation, provenance)",
    "bci_neuroai": "A BCI/NeuroAI translation project, separate from the worm work",
}

PRIMARY_ENDPOINTS: dict[str, str] = {
    "eight_states": "The released eight behavioural states",
    "binary_reversal": "Binary reversal onset/offset",
    "turn_direction": "Dorsal versus ventral turn",
    "continuous_kinematics": "A continuous kinematic variable",
    "other": "Another endpoint (specify below)",
}

AVAILABLE_DATA: dict[str, str] = {
    "sixth_worm": "A sixth held-out worm with the exact BunDLe-Net consistent embedding",
    "raw_traces_video": "Raw traces plus behaviour video for the three validation recordings",
    "randomized_intervention": "A randomized optogenetic/ablation intervention with matched sham",
    "independent_annotations": "An independent eight-behaviour video annotation",
    "none_yet": "None of the above yet",
}

SUCCESS_CRITERIA: dict[str, str] = {
    "positive_result": "A positive predictive or causal result",
    "rigorous_negative": "A rigorous negative result",
    "reusable_benchmark": "A reusable benchmark/protocol",
    "wet_lab_design": "A design for a wet-lab experiment",
    "working_demo": "A working product demo for the group",
}

TARGET_MODES: dict[str, str] = {
    "not_in_scope": "Not in scope for this project",
    "specified": "Specified below",
}

ANSWER_SOURCES: dict[str, str] = {
    "email": "Written email",
    "meeting_notes": "Meeting notes",
    "written_note": "Written note or document",
    "other": "Other written source",
}

IMPACT: dict[str, tuple[str, str]] = {
    "causal_mechanism": (
        "Randomized intervention + power + independent labels",
        "BCI lane; sixth worm as a goal in itself",
    ),
    "external_generalization": (
        "Sixth worm + independent annotations",
        "Intervention; BCI",
    ),
    "lab_workbench": (
        "Intake, QC, annotation UI, provenance",
        "No new scientific claim",
    ),
    "bci_neuroai": (
        "Separate EEG/BCI lane, ethics, BIDS",
        "Any mixing with worm claims",
    ),
}

def _clean(value: str) -> str:
    return value.strip()


@dataclass
class ScopeAnswers:
    """The six decisions plus provenance metadata.

    All decision fields default to ``None`` or empty so a new page cannot
    silently select an answer on Denis's behalf.
    """

    primary_object: str | None = None
    primary_endpoint: str | None = None
    endpoint_detail: str = ""
    available_data: tuple[str, ...] = field(default_factory=tuple)
    data_detail: str = ""
    success_criterion: str | None = None
    target_mode: str | None = None
    target_detail: str = ""
    stop_rule: str = ""
    budget_deadline: str = ""
    recorded_by: str = ""
    recorded_on: str = ""
    answer_source: str | None = None

    def decision_errors(self) -> list[str]:
        """Return every missing or contradictory decision."""
        errors: list[str] = []
        if self.primary_object is None:
            errors.append("Primary object is unanswered.")
        elif self.primary_object not in PRIMARY_OBJECTS:
            errors.append(f"Unknown primary object: {self.primary_object!r}.")

        if self.primary_endpoint is None:
            errors.append("Primary endpoint is unanswered.")
        elif self.primary_endpoint not in PRIMARY_ENDPOINTS:
            errors.append(f"Unknown primary endpoint: {self.primary_endpoint!r}.")
        elif self.primary_endpoint == "other" and not _clean(self.endpoint_detail):
            errors.append("Primary endpoint 'other' needs a written description.")

        if not self.available_data:
            errors.append("Available data is unanswered.")
        else:
            unknown = [item for item in self.available_data if item not in AVAILABLE_DATA]
            if unknown:
                errors.append(f"Unknown available-data item(s): {', '.join(unknown)}.")
            if "none_yet" in self.available_data and len(self.available_data) > 1:
                errors.append("'None yet' cannot be combined with another data item.")

        if self.success_criterion is None:
            errors.append("Definition of success is unanswered.")
        elif self.success_criterion not in SUCCESS_CRITERIA:
            errors.append(f"Unknown success criterion: {self.success_criterion!r}.")

        if self.target_mode is None:
            errors.append("Target/intervention decision is unanswered.")
        elif self.target_mode not in TARGET_MODES:
            errors.append(f"Unknown target mode: {self.target_mode!r}.")
        elif self.target_mode == "specified" and not _clean(self.target_detail):
            errors.append("A specified intervention needs a circuit and modality description.")

        if not _clean(self.stop_rule):
            errors.append("Stop rule is unanswered.")
        if not _clean(self.budget_deadline):
            errors.append("Budget/deadline is unanswered.")
        return errors

    def metadata_errors(self) -> list[str]:
        """Return provenance problems that block an auditable export."""
        errors: list[str] = []
        if not _clean(self.recorded_by):
            errors.append("Recorded by is missing.")
        if not _clean(self.recorded_on):
            errors.append("Recorded date is missing.")
        else:
            try:
                dt.date.fromisoformat(self.recorded_on)
            except ValueError:
                errors.append("Recorded date must be YYYY-MM-DD.")
        if self.answer_source is None:
            errors.append("Answer source is missing.")
        elif self.answer_source not in ANSWER_SOURCES:
            errors.append(f"Unknown answer source: {self.answer_source!r}.")
        return errors

    def validate(self) -> list[str]:
        """Return all decision and provenance errors."""
        return self.decision_errors() + self.metadata_errors()

    @property
    def is_complete(self) -> bool:
        """True only when all six decisions are answered consistently."""
        return not self.decision_errors()

    @property
    def is_exportable(self) -> bool:
        """True only when the decisions and their provenance are complete."""
        return not self.validate()

    def impact(self) -> tuple[str, str]:
        """Return the first workstream and deferred work for the chosen object."""
        if self.primary_object in IMPACT:
            return IMPACT[self.primary_object]
        return ("Blocked on G0: primary object unanswered", "All new workstreams")

    def status_label(self) -> str:
        """Return the only honest scope label for the current form state."""
        if not self.is_complete:
            return "G0 open — scope unconfirmed"
        if not self.is_exportable:
            return "Decisions recorded — provenance missing"
        return "Recorded, not signed"

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe transcription payload.

        Raises:
            ValueError: if any decision or provenance field is missing.
        """
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))
        assert self.primary_object is not None
        assert self.primary_endpoint is not None
        assert self.success_criterion is not None
        assert self.target_mode is not None
        assert self.answer_source is not None
        return {
            "schema": SCOPE_SCHEMA,
            "confirmation_status": CONFIRMATION_STATUS,
            "warning": (
                "This is an unconfirmed transcription of a reported answer, not a signature "
                "from Denis Kovacevic. The signed block in DENIS_REQUIREMENTS_BRIEF.md is "
                "the authoritative confirmation."
            ),
            "decisions": {
                "primary_object": {
                    "id": self.primary_object,
                    "label": PRIMARY_OBJECTS[self.primary_object],
                },
                "primary_endpoint": {
                    "id": self.primary_endpoint,
                    "label": PRIMARY_ENDPOINTS[self.primary_endpoint],
                    "detail": _clean(self.endpoint_detail),
                },
                "available_data": [
                    {"id": item, "label": AVAILABLE_DATA[item]} for item in self.available_data
                ],
                "data_detail": _clean(self.data_detail),
                "success_criterion": {
                    "id": self.success_criterion,
                    "label": SUCCESS_CRITERIA[self.success_criterion],
                },
                "target": {
                    "id": self.target_mode,
                    "label": TARGET_MODES[self.target_mode],
                    "detail": _clean(self.target_detail),
                },
                "stop_rule": _clean(self.stop_rule),
                "budget_deadline": _clean(self.budget_deadline),
            },
            "workstream": {
                "first": self.impact()[0],
                "deferred": self.impact()[1],
            },
            "metadata": {
                "recorded_by": _clean(self.recorded_by),
                "recorded_on": _clean(self.recorded_on),
                "answer_source": {
                    "id": self.answer_source,
                    "label": ANSWER_SOURCES[self.answer_source],
                },
            },
        }

    def to_markdown(self) -> str:
        """Return a portable, visibly unconfirmed transcription."""
        payload = self.to_payload()
        decisions = payload["decisions"]
        lines = [
            "# Denis scope response — UNCONFIRMED",
            "",
            "> **UNCONFIRMED — NOT SIGNED BY DENIS.** This is a transcription of a reported "
            "answer. Do not cite it as a confirmed requirement until the signed block in "
            "`DENIS_REQUIREMENTS_BRIEF.md` is returned.",
            "",
            f"- **Recorded by:** {payload['metadata']['recorded_by']}",
            f"- **Recorded on:** {payload['metadata']['recorded_on']}",
            f"- **Answer source:** {payload['metadata']['answer_source']['label']}",
            f"- **Schema:** `{payload['schema']}`",
            "",
            "## Decisions",
            "",
            f"- **Primary object:** {decisions['primary_object']['label']}",
            f"- **Primary endpoint:** {decisions['primary_endpoint']['label']}",
        ]
        endpoint_detail = decisions["primary_endpoint"]["detail"]
        if endpoint_detail:
            lines.append(f"  - Detail: {endpoint_detail}")
        data_labels = [item["label"] for item in decisions["available_data"]]
        lines.append(f"- **Available data:** {'; '.join(data_labels)}")
        data_detail = decisions["data_detail"]
        if data_detail:
            lines.append(f"  - Detail: {data_detail}")
        lines.extend(
            [
                f"- **Definition of success:** {decisions['success_criterion']['label']}",
                f"- **Target/intervention:** {decisions['target']['label']}",
            ]
        )
        target_detail = decisions["target"]["detail"]
        if target_detail:
            lines.append(f"  - Detail: {target_detail}")
        lines.extend(
            [
                f"- **Stop rule:** {decisions['stop_rule']}",
                f"- **Budget/deadline:** {decisions['budget_deadline']}",
                "",
                "## Workstream impact",
                "",
                f"- **First workstream:** {payload['workstream']['first']}",
                f"- **Deferred:** {payload['workstream']['deferred']}",
                "",
            ]
        )
        return "\n".join(lines)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ScopeAnswers:
        """Rebuild answers from :meth:`to_payload` output.

        Raises:
            ValueError: if the schema is unknown or a required field is absent.
        """
        if payload.get("schema") != SCOPE_SCHEMA:
            raise ValueError(f"Unsupported scope schema: {payload.get('schema')!r}")
        decisions = payload.get("decisions")
        metadata = payload.get("metadata")
        if not isinstance(decisions, dict) or not isinstance(metadata, dict):
            raise ValueError("Scope payload must contain decisions and metadata mappings.")
        try:
            available = decisions["available_data"]
            if not isinstance(available, list):
                raise TypeError("available_data must be a list")
            return cls(
                primary_object=decisions["primary_object"]["id"],
                primary_endpoint=decisions["primary_endpoint"]["id"],
                endpoint_detail=decisions["primary_endpoint"].get("detail", ""),
                available_data=tuple(item["id"] for item in available),
                data_detail=decisions.get("data_detail", ""),
                success_criterion=decisions["success_criterion"]["id"],
                target_mode=decisions["target"]["id"],
                target_detail=decisions["target"].get("detail", ""),
                stop_rule=decisions["stop_rule"],
                budget_deadline=decisions["budget_deadline"],
                recorded_by=metadata["recorded_by"],
                recorded_on=metadata["recorded_on"],
                answer_source=metadata["answer_source"]["id"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed scope payload: {exc}") from exc
