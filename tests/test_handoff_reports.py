"""Contract tests for the supervisor handoff pack."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/simulator/supervisor_handoff"
DOCS = ROOT / "docs"
JOINT = ROOT / "reports/simulator/joint_dynamics"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION \u2014 NOT CAUSALLY VALIDATED."
TITLE = (
    "NeuroBranch: Evaluating Generative Neural Population Dynamics and "
    "Model-Dependent Perturbations in C. elegans"
)

pytestmark = pytest.mark.skipif(not (OUT / "summary.json").exists(), reason="handoff reports not generated")


def test_validator_passes_with_hashes() -> None:
    script = ROOT / "scripts" / "validate_handoff_reports.py"
    spec = importlib.util.spec_from_file_location("validate_handoff_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main(check_hashes=True) == 0


def test_required_docs_carry_title_and_standing_labels() -> None:
    names = [
        "HANDOFF_EXECUTIVE_SUMMARY.md",
        "HANDOFF_RESEARCH_MEMO.md",
        "HANDOFF_PRESENTATION.md",
        "HANDOFF_SCIENTIFIC_CLAIMS.md",
        "HANDOFF_METHODS_AND_LIMITATIONS.md",
        "HANDOFF_RESEARCH_PROPOSALS.md",
        "HANDOFF_QUESTIONS_FOR_DENIS.md",
        "HANDOFF_DEMO_GUIDE.md",
        "HANDOFF_REPRODUCIBILITY.md",
    ]
    assert all((DOCS / name).exists() for name in names)
    for name in names:
        text = (DOCS / name).read_text(encoding="utf-8")
        assert WARNING in text
        assert "NOT_EVALUABLE" in text
    memo = (DOCS / "HANDOFF_RESEARCH_MEMO.md").read_text(encoding="utf-8")
    assert memo.replace("*C. elegans*", "C. elegans").count(f"# {TITLE}") == 1


def test_summary_numbers_match_frozen_tables() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(JOINT / "score_summary.csv").set_index("model")
    assert summary["title"] == TITLE
    assert summary["status"]["probabilistic"] == "PROBABILISTIC CALIBRATION NOT ESTABLISHED"
    assert summary["status"]["biological_validation"] == "NOT_EVALUABLE"
    assert summary["status"]["warning"] == WARNING
    for row in summary["scores"]:
        model = row["model"]
        assert abs(row["energy_h20"] - scores.loc[model, "energy_h20"]) < 1e-6
        assert abs(row["trajectory_h20"] - scores.loc[model, "trajectory_energy_h20"]) < 1e-6
    primary = summary["central_claim"]["primary"]
    paired = pd.read_csv(JOINT / "paired_scores.csv")
    row = paired[
        (paired["model"] == "J5_population_lead1_boot")
        & (paired["reference"] == "J1_independent_diag")
        & (paired["metric"] == "energy_h20")
    ].iloc[0]
    assert abs(primary["improvement"] - row["mean_improvement"]) < 1e-6
    assert primary["ci_low"] > 0
    assert len(summary["sessions"]) == 5
    assert sum(1 for item in summary["sessions"] if item["improvement"] > 0) == 4
    assert summary["gates"]["G8_biological_intervention"]["status"] == "NOT_EVALUABLE"


def test_timeline_and_claim_ledger_are_complete() -> None:
    timeline = pd.read_csv(OUT / "phase_timeline.csv")
    claims = pd.read_csv(OUT / "claim_ledger.csv")
    assert timeline["order"].tolist() == list(range(1, 13))
    assert len(timeline) == 12
    assert claims["id"].tolist() == [f"C{i}" for i in range(1, 19)]
    assert len(claims) == 18
    assert claims["status"].isin({"supported", "exploratory", "mixed", "technical", "negative", "withdrawn", "not_evaluable"}).all()


def test_pdfs_have_expected_lengths() -> None:
    def pages(path: Path) -> int:
        return len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))

    assert pages(OUT / "NeuroBranch_Handoff_Memo.pdf") == 2
    assert pages(OUT / "NeuroBranch_Handoff_Presentation.pdf") == 11


def test_manifest_hashes_and_compactness() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["root"] == "reports/simulator/supervisor_handoff"
    assert "trajectory_examples.npz" not in manifest["files"]
    for name, record in manifest["files"].items():
        path = OUT / name
        assert path.exists()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"]
        assert path.stat().st_size == record["bytes"]
    assert len(manifest["documents"]) == 9
    assert not any(
        path.suffix.lower() in {".mat", ".pt", ".pth", ".ckpt"} for path in OUT.rglob("*")
    )
