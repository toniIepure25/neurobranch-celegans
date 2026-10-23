"""Machine-readable provenance for experiments and derived artifacts.

The aim is that an independent operator can reconstruct what was run, on which
bytes, with which seed and configuration, and where every output lives.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from neurobranch.config import ExperimentConfig, git_state, resolve_path, sha256_file


def environment() -> dict[str, Any]:
    import scipy
    import sklearn
    import torch

    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
    }


def command_line() -> str:
    return " ".join([Path(sys.executable).name, *sys.argv])


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def split_manifest(datasets: Iterable[Any]) -> list[dict[str, Any]]:
    rows = []
    for d in datasets:
        row: dict[str, Any] = {"worm_id": d.worm_id, "n_frames": int(d.recording.n_frames), "splits": {}}
        for split, mask in d.masks.items():
            idx = np.flatnonzero(mask)
            row["splits"][split] = {
                "n_frames": int(len(idx)),
                "first_frame": int(idx[0]) if len(idx) else None,
                "last_frame": int(idx[-1]) if len(idx) else None,
                "fraction": float(len(idx) / len(mask)),
            }
        row["predictor_names"] = list(d.predictor_names)
        row["excluded_neurons"] = list(d.excluded_neurons)
        rows.append(row)
    return rows


def build_experiment_manifest(
    cfg: ExperimentConfig,
    datasets: Iterable[Any],
    *,
    kinds: Iterable[str],
    seeds: Iterable[int],
    output_dir: str | Path,
    artifacts: Iterable[str | Path] = (),
) -> dict[str, Any]:
    datasets = list(datasets)
    data_path = resolve_path(cfg.data.path)
    source_hashes = sorted({d.recording.source_sha256 for d in datasets})
    cfg_dict = cfg.to_dict()
    return {
        "schema": "neurobranch.experiment_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": cfg.name,
        "command": command_line(),
        "git": git_state(),
        "dataset": {
            "name": cfg.data.dataset,
            "path": str(data_path),
            "exists": data_path.exists(),
            "size_bytes": data_path.stat().st_size if data_path.exists() else None,
            "sha256": sha256_file(data_path) if data_path.exists() else None,
            "sha256_recorded_in_data": source_hashes,
            "license_note": "Kato et al. 2015; see docs/DATA_PROVENANCE.md",
        },
        "config": cfg_dict,
        "config_sha256": _hash_json(cfg_dict),
        "models": list(kinds),
        "seeds": list(seeds),
        "split": {
            "mode": cfg.split.mode,
            "purge_frames": cfg.split.purge_frames,
            "n_blocks": cfg.split.n_blocks,
            "test_block": cfg.split.test_block,
            "val_block": cfg.split.val_block,
            "per_worm": split_manifest(datasets),
        },
        "environment": environment(),
        "output_dir": str(resolve_path(output_dir)),
        "artifacts": [str(resolve_path(a)) for a in artifacts],
        "metric_version": "neurobranch.evaluation.metrics.v1",
        "notes": [
            "Primary blocked split is interpolation, not forecasting.",
            "Behaviour labels derive from AVA/SMD/RIB calcium and are not independent ground truth.",
            "No intervention on a living animal was performed.",
        ],
    }


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return p


def write_experiment_manifest(cfg, datasets, *, kinds, seeds, output_dir, artifacts=()) -> Path:
    payload = build_experiment_manifest(
        cfg, datasets, kinds=kinds, seeds=seeds, output_dir=output_dir, artifacts=artifacts
    )
    return write_manifest(resolve_path(output_dir) / "run_manifest.json", payload)


def git_is_available() -> bool:
    try:
        subprocess.check_output(["git", "--version"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False
