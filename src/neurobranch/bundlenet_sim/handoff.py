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
)

_FIXED_TIME = (1980, 1, 1, 0, 0, 0)
_PACKET_ROOT = "gap-closure"


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
