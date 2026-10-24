"""Write the Phase 5 reproducibility manifest."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase5" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True).stdout.strip()


def main() -> None:
    raw = ROOT / "data" / "raw" / "kramer2026" / "files"
    raw_files = sorted(p for p in raw.glob("*") if p.is_file()) if raw.exists() else []
    outputs = []
    for path in sorted((ROOT / "reports" / "phase5").rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            outputs.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    code = []
    for pattern in ("scripts/phase5_*.py", "src/neurobranch/behavior/*.py", "src/neurobranch/models/small_rnn.py"):
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                code.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    documentation = []
    for path in sorted((ROOT / "docs").glob("PHASE5_*.md")):
        documentation.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    dirty = "\n".join(
        line
        for line in git("status", "--porcelain").splitlines()
        if "reports/phase5/manifest.json" not in line
    )
    manifest = {
        "phase": 5,
        "status": {
            "exact_reference_reproduction": "BLOCKED_MISSING_INPUTS",
            "reconstruction": "PARTIAL_REPRODUCTION",
            "evaluation": "METHOD_RECONSTRUCTION",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "dirty_porcelain": dirty,
            "starting_head": "14a43710334393c2dd234e848fb445d4317a734b",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {"torch": __import__("torch").__version__, "numpy": __import__("numpy").__version__, "pandas": __import__("pandas").__version__},
        },
        "raw_data": {
            "root": "data/raw/kramer2026/files",
            "n_files": len(raw_files),
            "bytes": sum(p.stat().st_size for p in raw_files),
            "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in raw_files],
            "missing_public_files": ["03072024_N2_2.chxTracks.mat", "03072024_N2_3.chxTracks.mat"],
            "missing_required_aligned_csvs": ["SAAV_dorsal.csv", "SAAV_ventral.csv", "HC_dorsal.csv", "HC_ventral.csv"],
        },
        "commands": [
            "python scripts/phase5_protocol_compare.py --epochs 35 --regimes original grouped chronological external --modes behavior saav both",
            "python scripts/phase5_protocol_finalize.py",
            "python scripts/phase5_protocol_mode_deltas.py",
            "python scripts/phase5_online.py --epochs 40 --landmarks 0.6 1.2 2.4 3.6 4.8 7.2 --rnn-modes behavior behavior_saav behavior_circuit behavior_population",
            "python scripts/phase5_online_finalize.py",
            "python scripts/phase5_online_baselines.py",
            "python scripts/phase5_per_animal_summary.py",
            "python scripts/phase5_mechanisms.py",
            "python scripts/phase5_interventions.py",
            "python scripts/phase5_figures.py",
            "python scripts/phase5_power.py",
            "python scripts/phase5_write_manifest.py",
        ],
        "code": code,
        "documentation": documentation,
        "outputs": outputs,
        "notes": [
            "No JAX/Equinox; the GRU is an independent PyTorch architecture-compatible reimplementation.",
            "The four author-local aligned CSVs are absent; exact reproduction is blocked.",
            "No trial-level optogenetic table exists in the public archive; no virtual intervention was run.",
        ],
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(outputs)} outputs, {len(raw_files)} raw files)")


if __name__ == "__main__":
    main()
