"""End-to-end Stage-One run: gates, artifacts, determinism and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobranch.abstraction.stage1 import frozen_baseline_status, run_stage1, summary_line


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    out = tmp_path_factory.mktemp("stage1")
    return run_stage1(quick=True, out_dir=out), out


@pytest.mark.scientific
def test_all_frozen_gates_pass(run: tuple[dict, Path]) -> None:
    metrics, _ = run
    failed = [gate["gate"] for gate in metrics["gates"] if not gate["passed"]]
    assert not failed, f"gates failed: {failed}"
    assert metrics["verdict"]["technical_go"] is True
    assert summary_line(metrics).startswith("GO")


def test_verdict_states_what_is_not_claimed(run: tuple[dict, Path]) -> None:
    not_claimed = " ".join(run[0]["verdict"]["not_claimed"]).lower()
    assert "biological" in not_claimed
    assert "reproduc" in not_claimed


def test_positive_and_negative_controls_have_the_declared_signs(run: tuple[dict, Path]) -> None:
    exact = run[0]["exact"]
    assert exact["exactly_lumpable/positive"]["errors"]["passive"]["dynamic"] == 0.0
    assert exact["exactly_lumpable/behaviour_only"]["errors"]["passive"]["dynamic"] > 0.05
    assert exact["exactly_lumpable/intervention_only"]["errors"]["lesion"]["dynamic"] > 0.05


def test_reports_and_manifests_are_written(run: tuple[dict, Path]) -> None:
    _, out = run
    for name in ("metrics.json", "SUMMARY.md", "CONFORMANCE.md", "manifest.json", "manifest.sha256", "commutation_by_cell.csv"):
        path = out / name
        assert path.exists(), name
        assert path.stat().st_size > 0
    assert (out / "artifacts" / "ic1" / "contracts.json").exists()


@pytest.mark.scientific
def test_run_is_byte_identical_across_processes(tmp_path: Path) -> None:
    first = run_stage1(quick=True, out_dir=tmp_path / "a")
    second = run_stage1(quick=True, out_dir=tmp_path / "b")
    assert first == second
    payload_a = (tmp_path / "a" / "metrics.json").read_text(encoding="utf-8")
    payload_b = (tmp_path / "b" / "metrics.json").read_text(encoding="utf-8")
    assert payload_a == payload_b
    assert "created_utc" not in payload_a


@pytest.mark.scientific
def test_manifest_hashes_its_outputs(tmp_path: Path) -> None:
    run_stage1(quick=True, out_dir=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]
    assert "created_utc" in manifest
    digests = (tmp_path / "manifest.sha256").read_text(encoding="utf-8")
    for relative in manifest["outputs"]:
        assert relative.split("/")[-1] in digests


def test_conformance_report_does_not_claim_reproduction(tmp_path: Path) -> None:
    run_stage1(quick=True, out_dir=tmp_path)
    text = (tmp_path / "CONFORMANCE.md").read_text(encoding="utf-8")
    assert "not_reproduced" in text
    assert "julia" in text.lower()
    assert "Neither implementation ran" in text
    assert "What is deliberately not claimed" in text


def test_frozen_baseline_verifies() -> None:
    status = frozen_baseline_status()
    assert status["status"] == "ok"
    assert status["checked"] >= 40
    assert not status["mismatched"]
    assert not status["missing"]


def test_cli_stage1_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from neurobranch.cli.main import app

    monkeypatch.setattr("neurobranch.abstraction.stage1.REPORT_DIR", tmp_path)
    result = CliRunner().invoke(app, ["abstraction", "stage1", "--quick", "--seed", "5"])
    assert result.exit_code == 0, result.stdout
    assert "G1a_independent_reference" in result.stdout


def test_cli_contracts_refuses_privileged_comparison() -> None:
    from typer.testing import CliRunner

    from neurobranch.cli.main import app

    result = CliRunner().invoke(app, ["abstraction", "contracts", "--check"])
    assert result.exit_code == 0, result.stdout
    assert "ranking refused as designed" in result.stdout
