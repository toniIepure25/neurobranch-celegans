"""Write a reproducibility manifest for a completed Phase 8 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from neurobranch.config import PROJECT_ROOT, git_state, resolve_path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/phase8")
    args = parser.parse_args()
    out = resolve_path(args.output, PROJECT_ROOT)
    artifact_names = [
        "summary.json",
        "readout.json",
        "design.json",
        "metrics.csv",
        "predictions.parquet",
        "audit.json",
        "selection_log.jsonl",
        "README.md",
    ]
    artifacts = {}
    for name in artifact_names:
        path = out / name
        if path.exists():
            artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    models = {}
    model_dir = out / "models"
    if model_dir.exists():
        for path in sorted(model_dir.glob("*.pt")):
            models[path.name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    manifest = {
        "phase": 8,
        "name": "Adaptive Perturbation Twin",
        "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
        "git": git_state(PROJECT_ROOT),
        "artifacts": artifacts,
        "models": models,
        "preregistration": "docs/PHASE8_PREREGISTRATION.md",
        "config": "configs/phase8/apt.yaml",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
