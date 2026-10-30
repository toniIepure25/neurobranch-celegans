"""Write the Phase 12 manifest and external checksum."""

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
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "phase12")
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args(argv)
    if not args.reports.exists():
        print(f"reports directory not found: {args.reports}", file=sys.stderr)
        return 2

    summary_path = args.reports / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(args.reports.iterdir())
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }
    lockfile = ROOT / "uv.lock"
    status_lines = _git("status", "--porcelain").splitlines()
    dirty = any("reports/phase12/manifest" not in line for line in status_lines)
    input_record: dict[str, Any] = {
        "path": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "redistributed": False,
        "sha256": _sha256(INPUT) if INPUT.is_file() else None,
    }
    payload: dict[str, Any] = {
        "phase": "phase12",
        "artifacts": artifacts,
        "input": input_record,
        "parameters": {
            "seed": args.seed,
            "primary_coverage": summary.get("primary_coverage"),
            "n_eval": summary.get("n_eval"),
            "prior": summary.get("prior"),
            "status": summary.get("status"),
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "uv_lock_sha256": _sha256(lockfile) if lockfile.exists() else None,
        },
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": [
            "git_dirty ignores the manifest files being written and reflects all other worktree changes.",
            "Artifact hashes cover every file in this directory except manifest.json and manifest.sha256.",
            "The derived input table is referenced by hash and is not redistributed.",
        ],
    }
    manifest_path = args.reports / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = _sha256(manifest_path)
    (args.reports / "manifest.sha256").write_text(f"{checksum}  manifest.json\n", encoding="utf-8")
    print(f"wrote {manifest_path} ({len(artifacts)} artifacts, sha256 {checksum[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
