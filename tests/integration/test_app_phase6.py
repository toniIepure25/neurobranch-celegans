from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION"


def test_phase6_documents_exist_and_preserve_warning() -> None:
    names = [
        "PHASE6_PREREGISTRATION.md",
        "PHASE6_RESEARCH_PLAN.md",
        "PHASE6_DEVIATIONS.md",
        "PHASE6_RESULTS.md",
        "PHASE6_SUPERVISOR_BRIEF.md",
        "PHASE6_EXTERNAL_DATA.md",
    ]
    for name in names:
        path = ROOT / "docs" / name
        assert path.is_file(), name
        assert WARNING in path.read_text(encoding="utf-8"), name


def test_phase6_primary_artifacts_exist() -> None:
    paths = [
        "reports/phase6/preregistration.json",
        "reports/phase6/cro/metrics.csv",
        "reports/phase6/cro/predictions.parquet",
        "reports/phase6/cro/operator.npz",
        "reports/phase6/cro/uncertainty.npz",
        "reports/phase6/cro/audit.json",
        "reports/phase6/cro/profile_summary.csv",
        "reports/phase6/cro/connectivity/edge_metrics.csv",
        "reports/phase6/design/target_candidates.csv",
        "reports/phase6/randi_reconciliation.csv",
    ]
    for rel in paths:
        assert (ROOT / rel).is_file(), rel


def test_phase6_results_state_the_negative_and_calibration_limits() -> None:
    text = (ROOT / "docs" / "PHASE6_RESULTS.md").read_text(encoding="utf-8")
    assert "NOT SUPPORTED" in text
    assert "0.984" in text
    assert "causal" in text.lower()
    assert "cross-dataset" in text
