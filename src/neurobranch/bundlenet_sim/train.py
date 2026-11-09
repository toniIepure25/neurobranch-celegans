"""Fit and evaluate the BunDLe-Net latent simulator.

Usage
-----
    python -m neurobranch.bundlenet_sim.train --alpha 1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn

from .dataset import load_trajectories
from .dynamics import fit_models
from .evaluate import evaluate_leave_one_worm_out, save_summary
from .persistence import save_models


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_artifact_dir() -> Path:
    return repo_root() / "artifacts" / "bundlenet_sim"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path) -> dict[str, object]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout
        return {"sha": sha, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=1.0, help="ridge penalty")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    root = repo_root()
    artifact_dir = args.artifact_dir or default_artifact_dir()
    trajectories = load_trajectories(args.data_dir)
    summary = evaluate_leave_one_worm_out(trajectories, alpha=args.alpha)
    summary_path = save_summary(summary, artifact_dir / "metrics.json")

    final = fit_models(trajectories, alpha=args.alpha)
    model_path = save_models(final, artifact_dir / "model.npz")

    manifest = {
        "format": "neurobranch-bundlenet-sim-run-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m neurobranch.bundlenet_sim.train --alpha " + str(args.alpha),
        "git": git_state(root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "alpha": args.alpha,
        "n_worms": len(trajectories),
        "worm_ids": [t.worm_id for t in trajectories],
        "frames_per_worm": {str(t.worm_id): int(len(t.latent)) for t in trajectories},
        "files": {
            "model": file_record(model_path, root),
            "metrics": file_record(summary_path, root),
        },
        "source": "published BunDLe-Net consistent embeddings (NC-MCM release)",
        "scope": summary.to_dict()["scope"],
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary.to_dict(), indent=2))
    print(f"wrote {model_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
