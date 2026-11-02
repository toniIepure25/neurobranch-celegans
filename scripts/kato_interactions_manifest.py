"""Build a compact SHA-256 manifest for the Kato interaction audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "simulator" / "kato_interactions"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


def main() -> None:
    paths: list[Path] = []
    for folder in [
        ROOT / "src" / "neurobranch" / "kato_interactions",
        ROOT / "scripts",
        ROOT / "tests",
        ROOT / "docs",
        OUT / "figures",
    ]:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("_") or path.suffix in {".pyc", ".pkl"}:
                continue
            if path.name in {"manifest.json", "validation_report.json"}:
                continue
            if folder == ROOT / "scripts" and path.name not in {
                "kato_interactions_main.py",
                "kato_interactions_figures.py",
                "validate_kato_interactions.py",
                "kato_interactions_manifest.py",
            }:
                continue
            if folder == ROOT / "tests" and path.name != "test_kato_interactions.py":
                continue
            if folder == ROOT / "docs" and not (
                path.name == "mkdocs.yml" or path.name.startswith("KATO_INTERACTIONS_")
            ):
                continue
            paths.append(path)
    paths.extend(
        [
            ROOT / "app" / "handoff_demo.py",
            ROOT / "mkdocs.yml",
            ROOT / "reports" / "simulator" / "joint_dynamics" / "scores_by_landmark.csv",
            ROOT / "reports" / "simulator" / "joint_dynamics" / "gates.json",
        ]
    )
    seen: set[Path] = set()
    files = []
    for path in paths:
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    raw = ROOT / "data" / "raw" / "kato2015" / "WT_NoStim.mat"
    external = None
    if raw.exists():
        external = {
            "path": raw.relative_to(ROOT).as_posix(),
            "bytes": raw.stat().st_size,
            "sha256": sha256(raw),
            "usage": "input only; not redistributed or committed",
        }
    manifest = {
        "schema": "neurobranch.kato_interactions.manifest.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "dataset": "Kato et al. (2015) WT_NoStim only",
        "protocol": "docs/KATO_INTERACTIONS_PROTOCOL.md",
        "n_files": len(files),
        "files": files,
        "external_input": external,
        "excluded": ["_stage_a_cache.pkl", "validation_report.json", "manifest.json"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest written: {len(files)} files")


if __name__ == "__main__":
    main()
