"""Report-contract tests for the joint-dynamics experiment."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/simulator/joint_dynamics"
DOCS = ROOT / "docs"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."

pytestmark = pytest.mark.skipif(not (OUT / "manifest.json").exists(), reason="joint-dynamics reports not generated")


def test_report_validator_passes() -> None:
    script = ROOT / "scripts" / "validate_jointdyn_reports.py"
    spec = importlib.util.spec_from_file_location("validate_jointdyn_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main() == 0


def test_required_docs_exist_and_carry_warning() -> None:
    names = [
        "JOINTDYN_PROTOCOL.md",
        "JOINTDYN_REPRODUCTION.md",
        "JOINTDYN_POPULATION_INFORMATION.md",
        "JOINTDYN_INNOVATION_STRUCTURE.md",
        "JOINTDYN_CALIBRATION.md",
        "JOINTDYN_PARAMETER_UNCERTAINTY.md",
        "JOINTDYN_TRAJECTORY_FIDELITY.md",
        "JOINTDYN_RESULTS.md",
        "JOINTDYN_LIMITATIONS.md",
        "JOINTDYN_SUPERVISOR_BRIEF.md",
        "JOINTDYN_NEXT_EXPERIMENTS.md",
    ]
    assert all((DOCS / name).exists() for name in names)
    for name in ["JOINTDYN_PROTOCOL.md", "JOINTDYN_RESULTS.md", "JOINTDYN_LIMITATIONS.md", "JOINTDYN_SUPERVISOR_BRIEF.md"]:
        assert WARNING in (DOCS / name).read_text(encoding="utf-8")


def test_gate_record_is_frozen_and_honest() -> None:
    gates = json.loads((OUT / "gates.json").read_text(encoding="utf-8"))
    assert gates["G4_marginal_calibration"]["status"] == "FAIL"
    assert gates["G5_joint_trajectory"]["status"] == "FAIL"
    assert gates["G7_generative_credibility"]["status"] == "FAIL"
    assert gates["G8_biological_intervention"]["status"] == "NOT_EVALUABLE"
    assert gates["stop_label"] == "PROBABILISTIC CALIBRATION NOT ESTABLISHED"


def test_no_raw_recordings_or_weights_are_present() -> None:
    bad = [p for p in OUT.rglob("*") if p.suffix.lower() in {".mat", ".pt", ".pth", ".ckpt"}]
    assert not bad
    assert not (OUT / "trajectory_examples.npz").stat().st_size > 100 * 1024 * 1024
