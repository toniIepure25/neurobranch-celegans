"""Explicit raw-byte and canonical-text hashing.

The Phase 6 manifest mixed raw-byte hashes with files that Git normalizes from
CRLF to LF.  A verifier must never silently substitute one convention for the
other.  These helpers return records that state both the algorithm and the
normalization policy, while still accepting a bare legacy SHA256 string for
read-only compatibility.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

HashMode = Literal["raw", "canonical_text"]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_raw(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_text_bytes(path: str | Path) -> bytes:
    """UTF-8 text with CRLF and lone CR normalised to LF.

    This is deliberately not called a byte hash.  Binary artifacts must use
    ``raw``; the caller is responsible for declaring text files.
    """
    payload = Path(path).read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not UTF-8 text; use raw hashing") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def sha256_canonical_text(path: str | Path) -> str:
    return sha256_bytes(canonical_text_bytes(path))


def hash_record(path: str | Path, *, mode: HashMode = "raw") -> dict[str, Any]:
    p = Path(path)
    if mode == "raw":
        digest = sha256_raw(p)
    elif mode == "canonical_text":
        digest = sha256_canonical_text(p)
    else:
        raise ValueError(f"unknown hash mode {mode!r}")
    return {
        "path": p.name,
        "bytes": p.stat().st_size,
        "sha256": digest,
        "algorithm": "sha256",
        "normalization": mode,
    }


def verify_hash(
    path: str | Path,
    expected: str | dict[str, Any],
    *,
    mode: HashMode | None = None,
) -> tuple[bool, str]:
    """Verify one artifact without silently changing conventions.

    ``expected`` may be a bare legacy string (tries the explicitly requested
    mode, default raw) or a record from :func:`hash_record`.  Returns
    ``(ok, observed_mode_or_error)``.
    """
    p = Path(path)
    if isinstance(expected, dict):
        declared = expected.get("normalization")
        if declared not in {"raw", "canonical_text"}:
            return False, "missing_or_unknown_normalization"
        if mode is not None and mode != declared:
            return False, f"mode_mismatch:declared={declared}:requested={mode}"
        use: HashMode = declared
        digest = expected.get("sha256") or expected.get("sha256_raw")
        if not isinstance(digest, str):
            return False, "missing_digest"
    else:
        use = mode or "raw"
        digest = expected
    actual = sha256_raw(p) if use == "raw" else sha256_canonical_text(p)
    return actual == digest, use


def read_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def verify_manifest_records(
    manifest: dict[str, Any],
    *,
    root: str | Path = ".",
    allow_legacy_raw: bool = True,
) -> dict[str, Any]:
    """Verify a manifest whose entries may be records or legacy digest strings."""
    base = Path(root)
    results = []
    for entry in manifest.get("artifacts", []):
        target = base / entry["path"]
        expected = entry.get("sha256")
        if entry.get("normalization") in {"raw", "canonical_text"}:
            ok, mode = verify_hash(target, entry)
        elif isinstance(expected, str) and allow_legacy_raw:
            ok, mode = verify_hash(target, expected, mode="raw")
            mode = f"legacy_raw:{mode}"
        else:
            ok, mode = False, "unsupported_entry"
        results.append({"path": entry["path"], "ok": bool(ok), "mode": mode})
    return {
        "n_artifacts": len(results),
        "n_ok": sum(r["ok"] for r in results),
        "failures": [r for r in results if not r["ok"]],
        "results": results,
    }
