"""Write a portable manifest for the simulator reports and inputs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neurobranch.config import PROJECT_ROOT


def sha256(path: Path, *, text: bool = False) -> str:
    h = hashlib.sha256()
    if text:
        # Git stores these text artifacts with LF; hash the normalised bytes so
        # a fresh checkout verifies on Windows and Unix alike.
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return h.hexdigest()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return (
            subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> int:
    reports = PROJECT_ROOT / "reports" / "simulator"
    kato = PROJECT_ROOT / "data" / "raw" / "kato2015" / "WT_NoStim.mat"
    files = [
        "selection.json",
        "summary.json",
        "gates.json",
        "rollout_metrics.csv",
        "one_step_metrics.csv",
        "selected_model_test.csv",
        "latent_metrics.csv",
        "latent_stability.json",
        "branching_summary.json",
        "branching_events.csv",
        "perturbation_summary.json",
        "perturbation_specificity.csv",
        "perturbation_effects.csv",
        "perturbation_controls.csv",
    ]
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "input": {
            "kato_path": str(kato.relative_to(PROJECT_ROOT)),
            "kato_sha256": sha256(kato) if kato.exists() else None,
            "config_path": "configs/experiments/blocked_suite.yaml",
            "config_sha256": sha256(PROJECT_ROOT / "configs/experiments/blocked_suite.yaml", text=True),
            "split_path": "configs/splits/blocked.yaml",
            "split_sha256": sha256(PROJECT_ROOT / "configs/splits/blocked.yaml", text=True)
            if (PROJECT_ROOT / "configs/splits/blocked.yaml").exists()
            else None,
        },
        "reports": {
            name: {"sha256": sha256(reports / name, text=True)}
            for name in files
            if (reports / name).exists()
        },
        "commands": [
            "python scripts/simulator_fit.py",
            "python scripts/simulator_evaluate.py",
            "python scripts/simulator_latent.py",
            "python scripts/simulator_branching.py",
            "python scripts/simulator_perturb.py",
            "python scripts/simulator_figures.py",
            "python scripts/simulator_gates.py",
            "python scripts/validate_reports.py --reports reports/simulator --strict",
            "python scripts/simulator_manifest.py",
        ],
        "checkpoint_policy": (
            "Existing Phase 2 checkpoints are reused. --fit-missing never overwrites; latent "
            "encoder weights are local only and excluded by .gitignore/.gitattributes."
        ),
        "manifest_note": (
            "Generated before its own commit; git_dirty is true because the manifest itself "
            "was untracked. Text hashes are LF-normalised, so a clean checkout verifies."
        ),
        "warning": (
            "MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED. "
            "Kato 2015 is observational; Gate 6 is not_evaluable."
        ),
    }
    (reports / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote reports/simulator/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
