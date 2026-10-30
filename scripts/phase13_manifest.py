"""Write the Phase 13 artifact manifest and checksum."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"
PROTOCOL = ROOT / "research" / "phase13_protocol.md"
DEVIATIONS = ROOT / "research" / "phase13_deviations.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "phase13")
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args(argv)
    reports = args.reports.resolve()
    if not reports.exists():
        print(f"reports directory not found: {reports}", file=sys.stderr)
        return 2

    summary_path = reports / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(reports.iterdir())
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256", "validation.json"}
    }
    lockfile = ROOT / "uv.lock"
    # Generated report artifacts are the subject of this manifest, not code
    # provenance.  Exclude reports/ so a clean code commit can be certified even
    # while its derived tables are being written or staged.
    status_lines = _git("status", "--porcelain", "--", ".", ":(exclude)reports").splitlines()
    dirty = bool(status_lines)
    payload: dict[str, Any] = {
        "phase": "phase13",
        "decision": summary.get("decision"),
        "primary_model": summary.get("primary_model"),
        "quick": summary.get("quick"),
        "artifacts": artifacts,
        "input": {
            "path": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
            "redistributed": False,
            "sha256": _sha256(INPUT) if INPUT.is_file() else None,
        },
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(PROTOCOL) if PROTOCOL.is_file() else None,
        },
        "deviations": {
            "path": str(DEVIATIONS.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(DEVIATIONS) if DEVIATIONS.is_file() else None,
        },
        "parameters": {
            "seed": args.seed,
            "config": summary.get("config"),
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": __import__("numpy").__version__,
            "pandas": __import__("pandas").__version__,
            "uv_lock_sha256": _sha256(lockfile) if lockfile.exists() else None,
        },
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": [
            "git_dirty reflects code/worktree changes outside reports/; generated report files are excluded.",
            "Artifact hashes cover every file except manifest.json, manifest.sha256 and validation.json.",
            "The derived input table is referenced by hash and is not redistributed.",
            "Quick-mode outputs are not decision evidence.",
        ],
    }
    manifest_path = reports / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    checksum = _sha256(manifest_path)
    (reports / "manifest.sha256").write_text(f"{checksum}  manifest.json\n", encoding="utf-8", newline="\n")
    print(f"wrote {manifest_path} ({len(artifacts)} artifacts, sha256 {checksum[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
