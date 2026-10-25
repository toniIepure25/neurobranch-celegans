"""Write a provenance manifest for Phase 7 artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from neurobranch.config import PROJECT_ROOT, git_state, resolve_path

ARTEFACT_GLOBS = (
    "reports/phase7/**/*.json",
    "reports/phase7/**/*.csv",
    "reports/phase7/**/*.csv.gz",
    "reports/phase7/**/*.parquet",
    "reports/phase7/**/*.npz",
    "reports/phase7/**/*.md",
    "docs/PHASE7_*.md",
    "configs/phase7/*.yaml",
    "scripts/phase7_*.py",
    "src/neurobranch/active/*.py",
    "src/neurobranch/benchmark/conformal.py",
    "src/neurobranch/benchmark/metrics.py",
    "src/neurobranch/causal/features.py",
    "src/neurobranch/causal/response_operator.py",
    "app/streamlit_app.py",
    "tests/unit/test_phase7_active.py",
    "tests/integration/test_app_phase7.py",
    "tests/integration/test_app_phase3.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/phase7/manifest.json")
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
        "phase": 7,
        "created_by": "scripts/phase7_manifest.py",
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_state(PROJECT_ROOT),
        "n_files": len(files),
        "files": files,
        "notes": [
            "Raw calcium recordings and row-level Randi data are not redistributed.",
            "reports/phase7/audit.json contains the split and leakage checks.",
            "The active benchmark is pseudo-prospective, not a wet-lab closed loop.",
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out} with {len(files)} files")


if __name__ == "__main__":
    main()
