"""Write a machine-readable manifest for the Phase 3 prospective run."""

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
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "phase3"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def main() -> None:
    summary = json.loads((REPORTS / "prospective_summary.json").read_text(encoding="utf-8"))
    data_manifest = ROOT / "data" / "raw" / "baskoylu2026" / "manifest.json"
    labels = ROOT / "data" / "raw" / "baskoylu2026" / "neuropal_labels.tsv"
    artifacts = sorted(
        p for p in REPORTS.glob("prospective_*") if p.is_file() and p.name != "prospective_run_manifest.json"
    )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dataset": {
            "doi": "doi:10.5061/dryad.w9ghx3g4v",
            "manifest": str(data_manifest.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": sha256(data_manifest) if data_manifest.exists() else None,
            "labels_sha256": sha256(labels) if labels.exists() else None,
        },
        "analysis": {
            "target": "reversal onset in (t, t+H]",
            "horizons_s": [3, 6, 12],
            "primary_horizon_s": 6,
            "split": "leave-one-animal-out",
            "primary_model": "L2 logistic regression",
            "C": 0.1,
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 17,
        },
        "primary_result": {
            "delta_log_loss": summary["6s"]["aggregate"]["delta_behavior_neural"]["mean"],
            "ci95": summary["6s"]["aggregate"]["delta_behavior_neural"]["ci95"],
            "n_negative": summary["6s"]["aggregate"]["delta_behavior_neural"]["n_negative"],
            "n": summary["6s"]["aggregate"]["delta_behavior_neural"]["n"],
            "criterion_met": False,
        },
        "artifacts": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in artifacts
        },
    }
    out = REPORTS / "prospective_run_manifest.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    sys.exit(main())
