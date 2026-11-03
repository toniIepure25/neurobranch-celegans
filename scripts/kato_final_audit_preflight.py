"""Freeze the final-audit starting state without touching raw data."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jd_evaluate as JD  # noqa: E402

from neurobranch.adaptive.evaluation import evaluation_landmarks  # noqa: E402
from neurobranch.diagnostics.common import load_nostim  # noqa: E402
from neurobranch.preprocessing import split_starts  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    out = ROOT / "reports" / "simulator" / "kato_final_audit"
    out.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data" / "raw" / "kato2015" / "WT_NoStim.mat"
    cfg, data, recordings = load_nostim()
    payload: dict[str, object] = {
        "schema": "neurobranch.kato_final_audit.preflight.v1",
        "git": {
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": git("rev-parse", "HEAD"),
            "dirty_porcelain": git("status", "--porcelain=v1"),
            "recent": git("log", "-6", "--format=%H %s").splitlines(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "streamlit", "plotly", "pytest", "ruff")
            },
        },
        "dataset": {
            "path": str(raw.relative_to(ROOT)),
            "sha256": sha256(raw),
            "bytes": raw.stat().st_size,
            "sessions": len(data),
            "frames_total": int(sum(int(d.n_frames) for d in data)),
            "recording_names": [r.dataset_name for r in recordings],
        },
        "config": {
            "name": cfg.name,
            "dataset": cfg.data.dataset,
            "path": cfg.data.path,
            "exclude_behavior_neurons": cfg.data.exclude_behavior_neurons,
            "split": {
                "mode": cfg.split.mode,
                "train_fraction": cfg.split.train_fraction,
                "val_fraction": cfg.split.val_fraction,
                "test_fraction": cfg.split.test_fraction,
                "purge_frames": cfg.split.purge_frames,
                "seed": cfg.split.seed,
            },
        },
        "sessions": [],
        "seeds": {
            "historical_innovation": "7000 + session*10000 + landmark",
            "surrogate": "91000 + session*1000 + landmark*11 + draw",
            "bootstrap": "session_block_ci defaults seed=17, n_boot=1000",
            "pca": 17,
            "synthetic": [11, 23, 37],
        },
        "experiment": {
            "members": 100,
            "rollout_horizon": 100,
            "primary_horizon": 20,
            "stride": 10,
            "factor_ks": [1, 3, 5, 10],
            "primary_k": 10,
            "surrogate_draws": 6,
            "surrogate_methods": [
                "circular_full",
                "block_full",
                "common_preserved_residual_shift",
                "residual_preserved_factor_shift",
            ],
        },
    }
    for d in data:
        lm = evaluation_landmarks(d, horizon_max=100, stride=10)
        train_starts = split_starts(d, "train", history=2, horizon=20)
        pca, _, _ = JD.prepare(d)
        payload["sessions"].append(
            {
                "session": int(d.worm_id),
                "frames": int(d.n_frames),
                "neurons": int(d.X.shape[1]),
                "split_bounds": {k: [int(v[0]), int(v[1])] for k, v in d.split_bounds.items()},
                "train_frames_h20": int(len(train_starts)),
                "landmarks": int(len(lm)),
                "landmark_first": int(lm[0]),
                "landmark_last": int(lm[-1]),
                "pca_components": int(pca.n_components_),
                "pca_explained_variance_10": float(pca.explained_variance_ratio_.sum()),
            }
        )
    (out / "preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"preflight": str(out / "preflight.json"), "sessions": payload["sessions"]}, indent=2))


if __name__ == "__main__":
    main()
