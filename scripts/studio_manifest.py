"""Write the Scientific Studio handoff manifest with hashes.

The manifest is intentionally generated, not hand-edited. It records the exact
Git revision, the evidence/artifact hashes, the Studio documents, and the
commands needed to verify the handoff.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "reports" / "scientific_studio"
DOCS = ROOT / "docs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry(path: Path, role: str) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "role": role,
    }


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    trajectory_json = json.loads((STUDIO / "kato_trajectories.json").read_text(encoding="utf-8"))
    docs = sorted(DOCS.glob("STUDIO_*.md"))
    code = sorted((ROOT / "src" / "neurobranch" / "studio").rglob("*.py"))
    code += [
        ROOT / "app" / "neurobranch_studio.py",
        ROOT / "scripts" / "studio_build_trajectories.py",
        ROOT / "scripts" / "studio_browser_qa.py",
        ROOT / "scripts" / "studio_contact_sheet.py",
        ROOT / "scripts" / "studio_final_audit.py",
        ROOT / "scripts" / "studio_provenance_audit.py",
        ROOT / "scripts" / "studio_release_checks.py",
        ROOT / "scripts" / "master_review" / "independent_model_check.py",
    ]
    code += sorted((ROOT / "tests").glob("test_studio*.py"))
    code += [ROOT / "tests" / "test_innovation_horizon_contract.py"]
    screenshots = sorted((STUDIO / "screenshots").glob("*.png"))
    final_audit = sorted(
        path
        for path in (STUDIO / "final_audit").glob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".png", ".md", ".log", ".txt"}
    )
    release_evidence = sorted(
        path
        for path in (STUDIO / "release_candidate").glob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".png", ".md", ".txt"}
        and path.name != "integrity_report.json"
    )
    status_lines = [line for line in git("status", "--porcelain").splitlines() if line]
    generated = {
        "reports/scientific_studio/manifest.json",
        "reports/scientific_studio/release_candidate/integrity_report.json",
    }
    other_changes = [line for line in status_lines if line[3:].strip() not in generated]

    manifest = {
        "schema": "neurobranch.scientific_studio.manifest.v1",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": {
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "source_head": git("rev-parse", "HEAD"),
            "manifest_commit_note": (
                "The manifest is committed after this source HEAD. The final "
                "repository HEAD also adds the separately committed integrity report, "
                "which is excluded here to avoid a self-hash cycle."
            ),
            "worktree_clean_excluding_manifest": not other_changes,
        },
        "dataset": {
            "path": "data/raw/kato2015/WT_NoStim.mat",
            "bytes": 56_390_975,
            "sha256": "7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a",
            "source": "https://osf.io/2395t/",
            "doi": "10.1016/j.cell.2015.09.034",
        },
        "artifact": {
            "trajectory_npz": entry(STUDIO / "kato_trajectories.npz", "compact trajectories"),
            "trajectory_json": entry(STUDIO / "kato_trajectories.json", "manifest and validation"),
            "rollout_horizon": trajectory_json["rollout_horizon"],
            "evaluation_horizon": trajectory_json["evaluation_horizon"],
            "sessions": len(trajectory_json["sessions"]),
            "landmarks": sum(len(s["landmarks"]) for s in trajectory_json["sessions"]),
            "showcase_landmarks": sum(
                len(s["showcase_landmarks"]) for s in trajectory_json["sessions"]
            ),
            "score_validation": {
                "max_abs_difference": trajectory_json["score_validation"]["max_abs_difference"],
                "tolerance": trajectory_json["score_validation"]["tolerance"],
                "n_showcase_checks": trajectory_json["score_validation"]["n_showcase_checks"],
            },
        },
        "code": [entry(path, "studio implementation") for path in code],
        "docs": [entry(path, "studio documentation") for path in docs],
        "screenshots": [entry(path, "browser QA capture") for path in screenshots],
        "final_audit": [entry(path, "final audit evidence") for path in final_audit],
        "release_candidate": [
            entry(path, "release-candidate evidence") for path in release_evidence
        ],
        "source_of_truth": [
            "reports/master_review/dataset_identity_audit.json",
            "reports/master_review/statistical_audit.json",
            "reports/master_review/quality_gate.json",
            "reports/master_review/claims_ledger.json",
            "reports/master_review/corrections.json",
            "reports/simulator/kato_common_state/model_per_landmark.csv",
            "reports/simulator/kato_final_audit/uncertainty_summary.json",
            "reports/scientific_studio/kato_trajectories.npz",
        ],
        "commands": {
            "launch": "python -X utf8 -m streamlit run app/neurobranch_studio.py",
            "rebuild_artifact": "python -X utf8 scripts/studio_build_trajectories.py",
            "studio_tests": (
                "python -X utf8 -m pytest -q tests/test_studio_artifacts.py "
                "tests/test_studio_ui_contract.py tests/test_studio_app_smoke.py"
            ),
            "full_tests": "python -X utf8 -m pytest -q",
            "lint": "python -X utf8 -m ruff check src/neurobranch/studio app/neurobranch_studio.py",
            "docs": "python -X utf8 -m mkdocs build --strict",
            "manifest": "python -X utf8 scripts/studio_manifest.py",
            "independent_check": (
                "python -X utf8 scripts/master_review/independent_model_check.py "
                "--extended --out reports/scientific_studio/final_audit/independent_model_check.json"
            ),
            "final_audit": "python -X utf8 scripts/studio_final_audit.py",
            "browser_qa": "python -X utf8 scripts/studio_browser_qa.py",
            "contact_sheet": "python -X utf8 scripts/studio_contact_sheet.py",
            "provenance": "python -X utf8 scripts/studio_provenance_audit.py --pretty",
            "release_checks": "python -X utf8 scripts/studio_release_checks.py",
        },
        "verdict": (
            "READY_FOR_METHODOLOGICAL_SUPERVISOR_REVIEW; "
            "NOT_READY_FOR_CONFIRMATORY_OR_BIOLOGICAL_CLAIMS"
        ),
        "notes": [
            "The manifest is generated after the last content change.",
            "The 3D projection is PCA, not anatomy.",
            "The perturbation lab is a model counterfactual, not stimulation.",
            "The test tail was reused; the result is exploratory.",
            "The historical 20-step innovation discrepancy is resolved and covered by a negative control.",
            "Browser capture passed geometry/console checks; human visual inspection remains pending.",
        ],
    }
    out = STUDIO / "manifest.json"
    out.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out.relative_to(ROOT)}")
    print(
        f"source_head={manifest['git']['source_head']} "
        f"clean_excluding_manifest={manifest['git']['worktree_clean_excluding_manifest']}"
    )
    print(
        f"docs={len(docs)} code={len(code)} screenshots={len(screenshots)} "
        f"release_evidence={len(release_evidence)}"
    )


if __name__ == "__main__":
    main()
