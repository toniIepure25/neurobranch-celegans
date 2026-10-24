"""Write the Phase 4 manifest and a model-card/checkpoint note."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "phase4"
CODE_GLOBS = [
    "src/neurobranch/data/kramer.py",
    "src/neurobranch/behavior/turn_direction.py",
    "scripts/phase4_download_kramer.py",
    "scripts/phase4_audit_data.py",
    "scripts/phase4_independent_replication.py",
    "scripts/phase4_turn_direction.py",
    "scripts/phase4_dynamics.py",
    "scripts/phase4_controls.py",
    "scripts/phase4_ablation.py",
    "scripts/phase4_nonlinear.py",
    "scripts/phase4_make_figures.py",
    "scripts/phase4_write_manifest.py",
    "tests/data/test_kramer.py",
]
DOCS = [
    "docs/PHASE4_RESEARCH_PLAN.md",
    "docs/PHASE4_DATASET_AUDIT.md",
    "docs/PHASE4_INDEPENDENT_REPLICATION.md",
    "docs/PHASE4_TURN_DIRECTION_BENCHMARK.md",
    "docs/PHASE4_NEURAL_BRANCH_DYNAMICS.md",
    "docs/PHASE4_NEURON_SPECIFIC_ANALYSIS.md",
    "docs/PHASE4_INTERVENTION_EVIDENCE.md",
    "docs/PHASE4_RESULTS.md",
    "docs/PHASE4_SUPERVISOR_BRIEF.md",
    "docs/PHASE4_NEXT_EXPERIMENTS.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - manifest should still be written
        return f"unavailable: {exc}"


def main() -> None:
    checks = pd.read_csv(REPORT / "data_checksums.csv")
    report_files = sorted(p for p in REPORT.rglob("*") if p.is_file() and p.name != "phase4_manifest.json")
    code = {
        rel: {"sha256": sha256(ROOT / rel), "bytes": (ROOT / rel).stat().st_size}
        for rel in CODE_GLOBS
        if (ROOT / rel).exists()
    }
    docs = {
        rel: {"sha256": sha256(ROOT / rel), "bytes": (ROOT / rel).stat().st_size}
        for rel in DOCS
        if (ROOT / rel).exists()
    }
    artifacts = {
        str(p.relative_to(ROOT)).replace("\\", "/"): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in report_files
    }
    manifest = {
        "schema_version": 1,
        "phase": 4,
        "generated_utc": datetime.now(UTC).isoformat(),
        "git": {
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "raw_data": {
            "dataset_doi": "doi:10.5061/dryad.8sf7m0cz2",
            "version": 5,
            "manifest": "data/raw/kramer2026/manifest.json",
            "checksums": "reports/phase4/data_checksums.csv",
            "files_verified": int(len(checks)),
            "bytes_verified": int(checks["bytes"].sum()),
            "all_match": bool(checks["match"].all()),
            "note": "Two MATLAB .chxTracks.mat files were not downloaded; they are not used.",
        },
        "commands": [
            "python scripts/phase4_download_kramer.py",
            "python scripts/phase4_audit_data.py",
            "python scripts/phase4_independent_replication.py",
            "python scripts/phase4_turn_direction.py",
            "python scripts/phase4_dynamics.py",
            "python scripts/phase4_controls.py",
            "python scripts/phase4_ablation.py",
            "python scripts/phase4_nonlinear.py",
            "python scripts/phase4_make_figures.py",
            "python -m pytest -q",
            "python -m ruff check src tests scripts",
            "python scripts/phase4_write_manifest.py",
        ],
        "code": code,
        "docs": docs,
        "artifacts": artifacts,
        "model_checkpoint": {
            "status": "not_applicable",
            "reason": "The primary benchmark is refit leave-one-animal-out; no single deployed model or checkpoint exists.",
            "model_card": "reports/phase4/checkpoints/MODEL_SPEC.json",
        },
        "caveats": [
            "Not an exact reproduction of the paper: derived SAAV/HC CSVs and RNN were unavailable.",
            "No trial-level optogenetic recordings; no causal intervention claim.",
            "D/V labels inferred from orientation-corrected dv.",
            "Control/ablation permutation counts were reduced for runtime.",
        ],
    }
    (REPORT / "phase4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checkpoint = REPORT / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    spec = {
        "primary_model": "StandardScaler + LogisticRegression",
        "solver": "lbfgs",
        "C": 0.1,
        "max_iter": 1000,
        "class_weight": None,
        "evaluation": "leave-one-animal-out",
        "primary_window": "mid four-frame reversal segment, >=2 frames before offset",
        "primary_metric": "pooled out-of-fold AUROC",
        "uncertainty": "2000 animal-cluster bootstrap resamples, seed 17",
        "note": "No persisted weights: one model is fit per held-out animal.",
    }
    (checkpoint / "MODEL_SPEC.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (checkpoint / "README.md").write_text(
        "# Phase 4 model checkpoint note\n\n"
        "No trained checkpoint is committed.  The primary benchmark fits a separate "
        "StandardScaler + logistic regression for every left-out animal; a single "
        "checkpoint would either leak training animals or misrepresent the "
        "animal-disjoint protocol.  The exact model specification, seeds, folds and "
        "per-event predictions are in this directory and "
        "`reports/phase4/turn_direction_predictions.parquet`.\n",
        encoding="utf-8",
    )
    print(f"wrote {REPORT / 'phase4_manifest.json'} with {len(artifacts)} artifacts")


if __name__ == "__main__":
    main()
