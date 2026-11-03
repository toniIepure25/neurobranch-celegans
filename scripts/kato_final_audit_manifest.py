"""Write a machine-readable manifest for the Kato final audit."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports" / "simulator" / "kato_final_audit"
DATA = ROOT / "data" / "raw" / "kato2015" / "WT_NoStim.mat"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def add(files: list[dict], path: Path) -> None:
    if path.is_file() and path.name != "manifest.json":
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )


def main() -> int:
    files: list[dict] = []
    for path in sorted((AUDIT).rglob("*")):
        add(files, path)
    for name in (
        "KATO_FINAL_AUDIT_PROTOCOL.md",
        "KATO_FINAL_AUDIT_REPRODUCTION.md",
        "KATO_FINAL_AUDIT_COMPARABILITY.md",
        "KATO_FINAL_AUDIT_SIGN_CONVENTION.md",
        "KATO_FINAL_AUDIT_UNCERTAINTY.md",
        "KATO_FINAL_AUDIT_SCIENTIFIC_CLAIMS.md",
        "KATO_FINAL_AUDIT_RESULTS.md",
        "KATO_FINAL_AUDIT_SUPERVISOR_BRIEF.md",
    ):
        add(files, ROOT / "docs" / name)
    add(files, ROOT / ".gitattributes")
    add(files, ROOT / "pyproject.toml")
    for pattern in (
        "scripts/kato_common_state_main.py",
        "scripts/kato_common_state_postprocess.py",
        "scripts/kato_final_audit_*.py",
        "scripts/validate_kato_final_audit.py",
        "tests/test_kato_final_audit.py",
    ):
        for path in sorted(ROOT.glob(pattern)):
            add(files, path)
    for path in sorted((ROOT / "src" / "neurobranch" / "kato_common_state").glob("*.py")):
        add(files, path)
    add(files, ROOT / "src" / "neurobranch" / "kato_interactions" / "metrics.py")
    files = sorted({entry["path"]: entry for entry in files}.values(), key=lambda item: item["path"])

    preflight = json.loads((AUDIT / "preflight.json").read_text(encoding="utf-8"))
    full = json.loads((AUDIT / "full_rerun_status.json").read_text(encoding="utf-8"))
    comparison = json.loads((AUDIT / "reproduction_comparison.json").read_text(encoding="utf-8"))
    ledger = json.loads((AUDIT / "claims_ledger.json").read_text(encoding="utf-8"))
    decision = json.loads((AUDIT / "decision_record.json").read_text(encoding="utf-8"))
    status = git("status", "--porcelain")
    manifest = {
        "schema": "neurobranch.kato_final_audit.manifest.v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "audit_branch": git("branch", "--show-current"),
        "head_at_manifest_generation": git("rev-parse", "HEAD"),
        "starting_head": "358107a7a55470d48f8092b2a2dc7f704dbd37c1",
        "git_status_porcelain_at_generation": status.splitlines(),
        "dataset": {
            "name": "Kato et al. 2015 WT_NoStim",
            "path": DATA.relative_to(ROOT).as_posix(),
            "sha256": sha256(DATA),
            "bytes": DATA.stat().st_size,
            "scope": "WT_NoStim only; no WT_Stim or other datasets",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": preflight["environment"]["packages"],
            "pip_freeze": "reports/simulator/kato_final_audit/environment/pip_freeze.txt",
        },
        "design": {
            "sessions": 5,
            "frames": 15662,
            "native_dimensions": [103, 127, 123, 118, 121],
            "landmarks_h20": 175,
            "landmarks_by_session": [35, 35, 34, 38, 33],
            "split_bounds": {
                str(row["session"]): row["split_bounds"] for row in preflight["sessions"]
            },
            "seeds": {
                **preflight["seeds"],
                "audit_bootstrap_primary": "1700 + block; block 5 primary",
            },
            "config": preflight["config"],
        },
        "commands": {
            "clean_rerun": full["command"],
            "preflight": "python -X utf8 scripts/kato_final_audit_preflight.py",
            "comparison": "python -X utf8 scripts/kato_final_audit_compare.py",
            "uncertainty": "python -X utf8 scripts/kato_final_audit_uncertainty.py",
            "ten_factor": "python -X utf8 scripts/kato_final_audit_ten_factor.py",
            "comparability": "python -X utf8 scripts/kato_final_audit_comparability.py",
            "claims": "python -X utf8 scripts/kato_final_audit_claims.py",
            "figures": "python -X utf8 scripts/kato_final_audit_figures.py",
            "validator": "python -X utf8 scripts/validate_kato_final_audit.py",
            "tests": "python -X utf8 -m pytest tests/test_kato_final_audit.py -q",
        },
        "clean_rerun_status": comparison["status"],
        "clean_rerun": {
            "status": full["completed_normally"],
            "status_label": comparison["status"],
            "pid": full["pid"],
            "started": full["started"],
            "reported_elapsed_seconds": full["reported_elapsed_seconds"],
            "stdout_bytes": full["stdout_bytes"],
            "stderr_bytes": full["stderr_bytes"],
            "out_dir": full["out_dir"],
            "missing_comparison_files": comparison["missing_files"],
            "maximum_absolute_difference": max(
                item.get("max_abs_difference", 0.0) for item in comparison["comparisons"]
            ),
            "all_numeric_within_1e-8": comparison["all_numeric_within_1e-8"],
            "gate_status_match": next(
                item["status"] for item in comparison["comparisons"] if item["file"] == "gates.json"
            ),
        },
        "evidence": {
            "claims_counts": ledger["counts"],
            "no_confirmed_label": ledger["no_confirmed_label"],
            "decision": decision["decision"],
            "g4_numerical": decision["g4_numerical"],
            "g4_confirmatory": decision["g4_confirmatory"],
            "structural_identifiability": decision["structural_identifiability"],
            "biological_intervention_validity": decision["biological_intervention_validity"],
        },
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "known_omissions": [
            "Per-landmark metric rows are preserved, but raw ensemble trajectory arrays were not emitted by the official pipeline.",
            "The five sessions are repeated recordings; animal identity/independence is not documented.",
            "The historical run was interrupted and post-processed; the clean rerun is the reproduction evidence.",
        ],
        "frozen_artifacts": {
            "path": "reports/simulator/kato_common_state",
            "overwritten": False,
        },
    }
    (AUDIT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("file_count", "total_bytes", "audit_branch", "head_at_manifest_generation")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
