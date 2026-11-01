"""Dataset and artifact manifests for the context-dynamics branch.

Raw binary data are hashed as bytes. Text reports are hashed both raw and with
LF canonicalisation; the two values are never conflated.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, git_state
from neurobranch.reproduction.hashes import (
    canonical_text_bytes,
    sha256_bytes,
    sha256_raw,
)


def _dataset(path: Path, label: str, used_for: str) -> dict:
    rec = {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "used_for": used_for,
    }
    if path.exists():
        rec.update(
            bytes=path.stat().st_size,
            sha256_raw=sha256_raw(path),
            algorithm="sha256",
            normalization="raw_binary",
        )
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/context_dynamics")
    args = parser.parse_args(argv)
    out = Path(args.reports)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig.load(PROJECT_ROOT / "configs/experiments/forward_stress.yaml")
    nostim = Path(cfg.data.path)
    if not nostim.is_absolute():
        nostim = PROJECT_ROOT / nostim
    stim = Path(__import__("os").environ.get("TEMP", ".")) / "neurobranch_external" / "WT_Stim.mat"
    datasets = {
        "kato2015_nostim": _dataset(nostim, "Kato 2015 WT_NoStim", "Stages B, C, E, F, G; Phase 6 comparison"),
        "kato2015_stim": _dataset(stim, "Kato 2015 WT_Stim", "Stage D oxygen input audit and models"),
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "datasets": datasets,
                "raw_data_committed_to_git": False,
                "not_used": "Nichols 2017 is not used in this branch; no cross-dataset claim is made.",
                "git": git_state(),
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = []
    self_name = "artifact_manifest.json"
    skip_names = {self_name, "reproducibility_manifest.json"}
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip_names):
        rel = path.relative_to(out).as_posix()
        raw = sha256_raw(path)
        entry = {
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256_raw": raw,
            "algorithm": "sha256",
            "normalization": "raw",
        }
        try:
            canonical = sha256_bytes(canonical_text_bytes(path))
            entry["sha256_canonical_text"] = canonical
            entry["canonical_text_applicable"] = True
            entry["raw_equals_canonical"] = raw == canonical
        except (ValueError, UnicodeDecodeError):
            entry["sha256_canonical_text"] = None
            entry["canonical_text_applicable"] = False
            entry["raw_equals_canonical"] = None
        artifacts.append(entry)
    payload = {
        "schema_version": 1,
        "scope": str(out.relative_to(PROJECT_ROOT)),
        "n_artifacts": len(artifacts),
        "n_text_canonical_applicable": sum(a["canonical_text_applicable"] for a in artifacts),
        "n_raw_equals_canonical": sum(bool(a["raw_equals_canonical"]) for a in artifacts),
        "artifacts": artifacts,
        "git": git_state(),
    }
    (out / self_name).write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")

    environment = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "git": git_state(),
        "commands": [
            "python scripts/context_latent_baseline.py",
            "python scripts/context_compositional.py",
            "python scripts/context_oxygen_audit.py",
            "python scripts/context_oxygen_models.py",
            "python scripts/context_shared_dynamics.py",
            "python scripts/context_branching.py",
            "python scripts/context_perturbations.py",
            "python scripts/context_support_audit.py",
            "python scripts/context_figures.py",
            "python scripts/context_manifests.py",
        ],
        "seeds": {"model_fit": 17, "rollout": 17000, "bootstrap": 17},
        "notes": [
            "No test block was used for model selection.",
            "Kato Stim forecasts supply the measured schedule and are not forecasts of a future stimulus.",
            "The Kato Stim test block is a chronological tail, not animal-independent.",
        ],
    }
    (out / "reproducibility_manifest.json").write_text(
        json.dumps(environment, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(artifacts)} artifact records; {payload['n_raw_equals_canonical']} raw==canonical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
