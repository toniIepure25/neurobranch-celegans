"""Audit a returned signed scope block without authenticating the signer.

The digest binds the returned Markdown to the six recorded decisions. It does not
prove who signed the block, so every status returned here is explicit about the
remaining manual identity check.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

from neurobranch.bundlenet_sim.scope import ScopeAnswers

SIGNED_BRIEF_AUDIT_SCHEMA = "vienna-signed-brief-audit-v1"
IDENTITY_VERIFICATION = "manual_required"

INVALID_FORMAT = "invalid_format"
INCOMPLETE = "incomplete"
NO_DIGEST = "no_digest"
DIGEST_MISMATCH = "digest_mismatch"
BLOCKED_ON_TRANSCRIPTION = "blocked_on_transcription"
MATCHES_UNVERIFIED = "matches_unverified_identity"

STATUSES = (
    INVALID_FORMAT,
    INCOMPLETE,
    NO_DIGEST,
    DIGEST_MISMATCH,
    BLOCKED_ON_TRANSCRIPTION,
    MATCHES_UNVERIFIED,
)

_REQUIRED: tuple[tuple[str, str], ...] = (
    ("primary_object", "Primary object"),
    ("primary_endpoint", "Primary endpoint"),
    ("available_data", "Available data"),
    ("success_criterion", "Success criterion"),
    ("target_intervention", "Target / intervention (if any)"),
    ("stop_rule", "Stop rule"),
    ("budget_deadline", "Budget / deadline"),
    ("scope_digest", "Scope digest"),
    ("confirmed_by", "Confirmed by"),
    ("date", "Date"),
)

_ALIASES = {
    "primary_object": "primary_object",
    "primary_endpoint": "primary_endpoint",
    "available_data": "available_data",
    "success_criterion": "success_criterion",
    "target_intervention_if_any": "target_intervention",
    "target_intervention": "target_intervention",
    "target": "target_intervention",
    "stop_rule": "stop_rule",
    "budget_deadline": "budget_deadline",
    "budget_and_deadline": "budget_deadline",
    "scope_digest": "scope_digest",
    "confirmed_by": "confirmed_by",
    "date": "date",
}

_FENCE = re.compile(r"```(?:text)?[ \t]*\n(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9 /()\-]*?)\s*:\s*(.*)$")
_SAME_LINE_DATE = re.compile(r"\s{2,}Date\s*:\s*(.*)$", re.IGNORECASE)


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _clean(value: str) -> str:
    value = value.strip().strip("`").strip()
    if re.fullmatch(r"[_\-\s]+", value):
        return ""
    return value


@dataclass(frozen=True)
class SignedBrief:
    """The human-readable fields from a returned confirmation block."""

    primary_object: str = ""
    primary_endpoint: str = ""
    available_data: str = ""
    success_criterion: str = ""
    target_intervention: str = ""
    stop_rule: str = ""
    budget_deadline: str = ""
    scope_digest: str = ""
    confirmed_by: str = ""
    date: str = ""
    format_errors: tuple[str, ...] = ()

    @classmethod
    def parse_markdown(cls, text: str) -> SignedBrief:
        """Parse the last fenced block that contains a ``Primary object:`` line."""
        chosen: str | None = None
        for match in _FENCE.finditer(text):
            body = match.group("body")
            if re.search(r"^\s*Primary object\s*:", body, re.MULTILINE | re.IGNORECASE):
                chosen = body
        if chosen is None:
            return cls(
                format_errors=(
                    "No fenced confirmation block containing 'Primary object:' was found.",
                )
            )

        values: dict[str, str] = {}
        errors: list[str] = []
        for line in chosen.splitlines():
            match = _LINE.match(line.strip())
            if match is None:
                continue
            raw_key, raw_value = match.groups()
            key = _ALIASES.get(_normalise_key(raw_key))
            if key is None:
                continue
            value = raw_value
            if key == "confirmed_by":
                same_line = _SAME_LINE_DATE.search(value)
                if same_line is not None:
                    if "date" in values:
                        errors.append("Duplicate field: Date.")
                    else:
                        values["date"] = _clean(same_line.group(1))
                    value = value[: same_line.start()]
            if key in values:
                errors.append(f"Duplicate field: {raw_key.strip()}.")
                continue
            values[key] = _clean(value)
            if key == "scope_digest":
                values[key] = values[key].lower()

        kwargs = {key: values.get(key, "") for key, _ in _REQUIRED}
        return cls(**kwargs, format_errors=tuple(errors))

    def missing_fields(self) -> tuple[str, ...]:
        """Return canonical labels for empty or placeholder-only fields."""
        return tuple(label for key, label in _REQUIRED if not getattr(self, key))

    def validation_errors(self) -> tuple[str, ...]:
        """Return format, completeness, date and digest errors."""
        errors = list(self.format_errors)
        missing = self.missing_fields()
        for label in missing:
            errors.append(f"{label} is missing.")
        if self.scope_digest and not _DIGEST.match(self.scope_digest):
            errors.append("Scope digest must be 64 lowercase hexadecimal characters.")
        if self.date:
            try:
                dt.date.fromisoformat(self.date)
            except ValueError:
                errors.append("Date must be YYYY-MM-DD.")
        return tuple(errors)

    def compare(self, answers: ScopeAnswers) -> SignedBriefComparison:
        """Compare the signed block with the current transcription."""
        current_digest = answers.scope_digest() if answers.is_exportable else ""
        if not answers.is_exportable:
            return SignedBriefComparison(
                status=BLOCKED_ON_TRANSCRIPTION,
                missing_fields=self.missing_fields(),
                signed_digest=self.scope_digest,
                current_digest=current_digest,
                digest_match=False,
                note=(
                    "Complete the six decisions and provenance before checking a signed brief."
                ),
            )
        if self.format_errors:
            return SignedBriefComparison(
                status=INVALID_FORMAT,
                missing_fields=self.missing_fields(),
                signed_digest=self.scope_digest,
                current_digest=current_digest,
                digest_match=False,
                note="; ".join(self.format_errors),
            )

        missing = self.missing_fields()
        if any(label != "Scope digest" for label in missing):
            return SignedBriefComparison(
                status=INCOMPLETE,
                missing_fields=missing,
                signed_digest=self.scope_digest,
                current_digest=current_digest,
                digest_match=False,
                note="The signed block is missing required fields.",
            )
        if not self.scope_digest:
            return SignedBriefComparison(
                status=NO_DIGEST,
                missing_fields=missing,
                signed_digest="",
                current_digest=current_digest,
                digest_match=False,
                note=(
                    "The signed block has no scope digest, so it cannot be bound to the "
                    "current transcription."
                ),
            )
        if not _DIGEST.match(self.scope_digest):
            return SignedBriefComparison(
                status=INCOMPLETE,
                missing_fields=missing,
                signed_digest=self.scope_digest,
                current_digest=current_digest,
                digest_match=False,
                note="The scope digest is malformed.",
            )
        if self.date:
            try:
                dt.date.fromisoformat(self.date)
            except ValueError:
                return SignedBriefComparison(
                    status=INCOMPLETE,
                    missing_fields=missing,
                    signed_digest=self.scope_digest,
                    current_digest=current_digest,
                    digest_match=False,
                    note="The signed date is not a valid YYYY-MM-DD date.",
                )
        match = self.scope_digest == current_digest
        if not match:
            return SignedBriefComparison(
                status=DIGEST_MISMATCH,
                missing_fields=missing,
                signed_digest=self.scope_digest,
                current_digest=current_digest,
                digest_match=False,
                note=(
                    "The signed digest does not match the current transcription. Do not treat "
                    "this block as confirmation of these decisions."
                ),
            )
        return SignedBriefComparison(
            status=MATCHES_UNVERIFIED,
            missing_fields=missing,
            signed_digest=self.scope_digest,
            current_digest=current_digest,
            digest_match=True,
            note=(
                "The fields are complete and the digest matches. The signer identity is not "
                "verified; a human must check Confirmed by before relying on the block."
            ),
        )

    def to_audit_payload(self, comparison: SignedBriefComparison) -> dict[str, Any]:
        """Return a JSON-safe audit record, never a claim of authentication."""
        return {
            "schema": SIGNED_BRIEF_AUDIT_SCHEMA,
            "status": comparison.status,
            "identity_verification": IDENTITY_VERIFICATION,
            "signed_digest": comparison.signed_digest,
            "current_digest": comparison.current_digest,
            "digest_match": comparison.digest_match,
            "missing_fields": list(comparison.missing_fields),
            "confirmed_by": self.confirmed_by,
            "date": self.date,
            "note": comparison.note,
        }

    def to_audit_markdown(self, comparison: SignedBriefComparison) -> str:
        """Return a portable audit record with the identity caveat in the header."""
        lines = [
            "# Signed brief audit — identity not verified",
            "",
            "> **Not an authentication result.** A digest match proves file consistency, not who",
            "> signed. A human must check `Confirmed by` against the returned document.",
            "",
            f"- **Status:** `{comparison.status}`",
            f"- **Identity verification:** `{IDENTITY_VERIFICATION}`",
            f"- **Confirmed by (reported):** {self.confirmed_by or '—'}",
            f"- **Date (reported):** {self.date or '—'}",
            f"- **Signed digest:** `{comparison.signed_digest or '—'}`",
            f"- **Current digest:** `{comparison.current_digest or '—'}`",
            f"- **Digest match:** {comparison.digest_match}",
            f"- **Note:** {comparison.note}",
        ]
        if comparison.missing_fields:
            lines.extend(
                [
                    "",
                    "## Missing fields",
                    "",
                    *(f"- {label}" for label in comparison.missing_fields),
                ]
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class SignedBriefComparison:
    """The result of comparing a signed block with a transcription."""

    status: str
    missing_fields: tuple[str, ...]
    signed_digest: str
    current_digest: str
    digest_match: bool
    note: str
