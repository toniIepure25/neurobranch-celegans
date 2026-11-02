"""Build a byte-level SHA256 manifest for Kato common-state artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "simulator" / "kato_common_state"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    files = {
        path.relative_to(OUT).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    payload = {
        "schema": "kato_common_state_manifest_v1",
        "git_head": head(),
        "files": files,
        "n_files": len(files),
    }
    (OUT / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"manifest: {len(files)} files")


if __name__ == "__main__":
    main()
