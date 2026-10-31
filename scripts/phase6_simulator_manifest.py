"""Build the Phase 6 simulator reproducibility manifest (collision-safe)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, sha256_file


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/phase6")
    parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "figures/simulator/phase6")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    args = parser.parse_args(argv)
    cfg = ExperimentConfig.load(args.config)
    reports, figures = Path(args.reports), Path(args.figures)
    artifacts = {}
    for base in (reports, figures):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.name != "simulator_manifest.json" and path.suffix in {
                ".csv", ".json", ".png", ".md"
            }:
                artifacts[str(path.relative_to(PROJECT_ROOT))] = {
                    "sha256": _sha(path),
                    "bytes": path.stat().st_size,
                }
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "matplotlib", "streamlit"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "schema_version": 1,
        "phase": 6,
        "evidence_tier": "exploratory",
        "test_block_reused": True,
        "git": {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git("rev-parse", "HEAD"),
            "start_head": "8d945f9ad755aabb4f0a30e93c2f24a8928a0614",
            "recent_commits": _git("log", "--oneline", "-12").splitlines(),
            "status_porcelain": _git("status", "--porcelain").splitlines(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
        },
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256_file(args.config),
        "data_sha256": sha256_file(cfg.data.path),
        "protocol_sha256": sha256_file(PROJECT_ROOT / "docs/PHASE6_RESEARCH_PROTOCOL.md"),
        "commands": [
            "python scripts/phase6_audit.py",
            "python scripts/phase6_select.py --families var stochastic switching",
            "python scripts/phase6_evaluate.py",
            "python scripts/phase6_branching.py --rollouts 80",
            "python scripts/phase6_perturbation.py",
            "python scripts/phase6_support_sensitivity.py",
            "python scripts/phase6_gates.py",
            "python scripts/phase6_cards.py",
            "python scripts/phase6_figures.py",
            "python -m pytest tests/simulator -q",
            "python -m ruff check src scripts app tests",
        ],
        "artifacts": artifacts,
        "limitations": [
            "test block 2 was inspected in Phase 5; no confirmatory claim",
            "Kato is unstimulated WT; no matched empirical intervention",
            "switching VAR is diagonal and failed validation skill",
            "no model beats persistence on the validation block",
        ],
        "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
    }
    target = reports / "simulator_manifest.json"
    target.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(target), "n_artifacts": len(artifacts), "head": payload["git"]["head"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
