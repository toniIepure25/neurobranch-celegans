"""Write a provenance manifest for Phase 6 artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from neurobranch.config import PROJECT_ROOT, git_state, resolve_path

ARTEFACT_GLOBS = (
    "reports/phase6/**/*.json",
    "reports/phase6/**/*.csv",
    "reports/phase6/**/*.csv.gz",
    "reports/phase6/**/*.parquet",
    "reports/phase6/**/*.npz",
    "docs/PHASE6_*.md",
    "configs/phase6/*.yaml",
    "scripts/phase6_*.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/phase6/manifest.json")
    args = parser.parse_args()
    out = resolve_path(args.output, PROJECT_ROOT)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for pattern in ARTEFACT_GLOBS:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            if not path.is_file() or path == out:
                continue
            files.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    payload = {
        "phase": 6,
        "created_by": "scripts/phase6_manifest.py",
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_state(PROJECT_ROOT),
        "n_files": len(files),
        "files": files,
        "notes": [
            "Raw calcium recordings and row-level Randi data are not redistributed.",
            "reports/phase6/cro/audit.json contains the benchmark-specific provenance.",
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out} with {len(files)} files")


if __name__ == "__main__":
    main()
