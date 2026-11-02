"""Reproducibility manifests for adaptive-dynamics artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path, *, name: str = "manifest.json") -> dict[str, Any]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != name)
    payload = {
        "root": str(root.name),
        "n_files": len(files),
        "files": {
            str(p.relative_to(root)).replace("\\", "/"): {
                "sha256": sha256_file(p),
                "bytes": int(p.stat().st_size),
            }
            for p in files
        },
    }
    (root / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_manifest(root: Path, *, name: str = "manifest.json") -> tuple[bool, list[str]]:
    root = Path(root)
    manifest = json.loads((root / name).read_text(encoding="utf-8"))
    problems = []
    for rel, record in manifest.get("files", {}).items():
        path = root / rel
        if not path.exists():
            problems.append(f"missing: {rel}")
        elif sha256_file(path) != record.get("sha256"):
            problems.append(f"hash mismatch: {rel}")
    return not problems, problems
