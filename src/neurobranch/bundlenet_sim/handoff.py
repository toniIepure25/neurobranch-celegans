"""Build a portable ZIP of the gap-closure handoff documents.

The packet is a transfer convenience, not evidence that any scientific gap is
closed.  It is deterministic so that the same repository state produces the same
bytes and can be hashed by the recipient.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

from neurobranch.bundlenet_sim.scope import CONFIRMATION_STATUS, SCOPE_SCHEMA

PACKET_FILES: tuple[str, ...] = (
    "docs/DENIS_REQUIREMENTS_BRIEF.md",
    "docs/DENIS_PUBLIC_SOURCE_AUDIT.md",
    "docs/GAP_CLOSURE_MATRIX.md",
    "docs/DATA_REQUEST_PACKAGE.md",
    "docs/ANNOTATION_PROTOCOL.md",
    "docs/INTERVENTION_POWER_PLAN.md",
    "docs/PUBLIC_DATA_GAP_MEMO.md",
    "reports/vienna_workbench/intervention_power.json",
    "schemas/cohort_manifest.schema.json",
    "schemas/scope_response.schema.json",
    "schemas/signed_brief_audit.schema.json",
)

DECISION_PACKET_FILES: tuple[str, ...] = (
    "docs/DENIS_REQUIREMENTS_BRIEF.md",
    "docs/DENIS_PUBLIC_SOURCE_AUDIT.md",
    "docs/DATA_REQUEST_PACKAGE.md",
    "docs/GAP_CLOSURE_MATRIX.md",
    "schemas/scope_response.schema.json",
    "schemas/signed_brief_audit.schema.json",
)

_FIXED_TIME = (1980, 1, 1, 0, 0, 0)
_PACKET_ROOT = "gap-closure"
_DECISION_ROOT = "decision-packet"


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def build_gap_packet(root: str | Path) -> bytes:
    """Return a deterministic ZIP containing the handoff documents and hashes."""
    root = Path(root).resolve()
    entries: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes]] = []
    for relative in PACKET_FILES:
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"packet file escapes repository: {relative}") from exc
        if not source.is_file():
            raise FileNotFoundError(f"missing packet file: {relative}")
        data = source.read_bytes()
        payloads.append((relative, data))
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )

    manifest = {
        "format": "vienna-gap-closure-packet-v1",
        "caution": "These documents describe open gaps; they do not close them.",
        "files": entries,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, data in payloads:
            archive.writestr(_zip_info(f"{_PACKET_ROOT}/{relative}"), data)
        archive.writestr(
            _zip_info("MANIFEST.json"),
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    return buffer.getvalue()


def _validate_decision_payload(
    payload: dict[str, object],
    scope_digest: str,
) -> None:
    if payload.get("schema") != SCOPE_SCHEMA:
        raise ValueError("decision packet payload is not a scope response")
    if payload.get("confirmation_status") != CONFIRMATION_STATUS:
        raise ValueError("decision packet payload is not an unconfirmed transcription")
    if not isinstance(scope_digest, str) or len(scope_digest) != 64:
        raise ValueError("decision packet digest must be a 64-character SHA-256 hex string")
    try:
        int(scope_digest, 16)
    except ValueError as exc:
        raise ValueError("decision packet digest must be hexadecimal") from exc
    if payload.get("scope_digest") != scope_digest:
        raise ValueError("decision packet digest does not match the payload")


def build_decision_packet(
    root: str | Path,
    *,
    contract_markdown: str,
    answers_markdown: str,
    payload: dict[str, object],
    scope_digest: str,
) -> bytes:
    """Build a deterministic ZIP for a reviewable, unconfirmed decision handoff.

    The packet is a transfer convenience.  It is not a signature, an identity
    check, or evidence that any scientific gap is closed.
    """
    _validate_decision_payload(payload, scope_digest)
    root = Path(root).resolve()
    entries: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes]] = []

    for relative in DECISION_PACKET_FILES:
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"decision packet file escapes repository: {relative}") from exc
        if not source.is_file():
            raise FileNotFoundError(f"missing decision packet file: {relative}")
        data = source.read_bytes()
        archive_name = f"{_DECISION_ROOT}/{relative}"
        payloads.append((archive_name, data))
        entries.append(
            {
                "path": archive_name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )

    dynamic = {
        "denis_scope_contract.md": contract_markdown,
        "denis_scope_response.json": json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        "denis_scope_response.md": answers_markdown,
    }
    for name, text in dynamic.items():
        data = text.encode("utf-8")
        archive_name = f"{_DECISION_ROOT}/{name}"
        payloads.append((archive_name, data))
        entries.append(
            {
                "path": archive_name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )

    manifest = {
        "format": "vienna-decision-packet-v1",
        "confirmation_status": CONFIRMATION_STATUS,
        "scope_digest": scope_digest,
        "caution": (
            "This packet records an unconfirmed transcription. It is not a signature, "
            "not evidence, and not a completed decision."
        ),
        "files": entries,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for archive_name, data in payloads:
            archive.writestr(_zip_info(archive_name), data)
        archive.writestr(
            _zip_info("MANIFEST.json"),
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    return buffer.getvalue()
