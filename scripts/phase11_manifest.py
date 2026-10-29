"""Write a complete Phase 11 manifest and its external checksum.

The manifest includes every generated artifact, the input hash, Git revision and
dirty state, the lockfile hash and the seed.  ``manifest.sha256`` is kept
outside the manifest so the manifest itself is verifiable.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "phase11")
    parser.add_argument("--seed", type=int, default=11)
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
    dirty = any("reports/phase11/manifest" not in line for line in status_lines)
    payload: dict[str, Any] = {
        "phase": "phase11",
        "artifacts": artifacts,
        "input": summary.get("input", {}),
        "parameters": {"seed": args.seed, **summary.get("parameters", {})},
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
