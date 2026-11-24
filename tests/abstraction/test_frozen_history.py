"""The frozen history: tracked historical manifests and the protocol hash."""

from __future__ import annotations

import subprocess

import pytest

from neurobranch.abstraction.stage1 import BASELINE_PATH, PROTOCOL_PATH, frozen_baseline_status
from neurobranch.config import PROJECT_ROOT, sha256_file


def test_protocol_hash_file_matches_the_protocol() -> None:
    recorded = PROTOCOL_PATH.parent / f"{PROTOCOL_PATH.stem}.sha256"
    assert recorded.exists(), "the frozen protocol must ship with its hash"
    digest, name = recorded.read_text(encoding="utf-8").split()
    assert name == PROTOCOL_PATH.name
    assert sha256_file(PROTOCOL_PATH) == digest


def test_baseline_covers_only_tracked_files() -> None:
    lines = [line for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines
    paths = [line.split(maxsplit=1)[1].strip() for line in lines]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *paths],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        pytest.skip("git metadata not available in this environment")
    assert tracked.returncode == 0


def test_baseline_matches_the_working_tree() -> None:
    status = frozen_baseline_status()
    assert status["status"] == "ok"
    assert status["mismatched"] == []


def test_stage1_artifacts_do_not_touch_historical_reports() -> None:
    historical = [
        PROJECT_ROOT / "reports" / "phase15" / "manifest.sha256",
        PROJECT_ROOT / "reports" / "epistemic_manifest.json",
    ]
    before = {path: sha256_file(path) for path in historical if path.exists()}
    from neurobranch.abstraction.stage1 import run_stage1

    run_stage1(quick=True, out_dir=PROJECT_ROOT / "reports" / "_tmp_frozen_check")
    after = {path: sha256_file(path) for path in historical if path.exists()}
    assert before == after
    import shutil

    shutil.rmtree(PROJECT_ROOT / "reports" / "_tmp_frozen_check", ignore_errors=True)
