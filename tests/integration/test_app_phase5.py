from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def test_phase5_required_documents_exist() -> None:
    names = [
        "PHASE5_RESEARCH_PLAN.md",
        "PHASE5_KRAMER_REPRODUCTION.md",
        "PHASE5_PROTOCOL_COMPARISON.md",
        "PHASE5_ONLINE_DECODING.md",
        "PHASE5_NEURAL_MECHANISMS.md",
        "PHASE5_INTERVENTION_EVIDENCE.md",
        "PHASE5_RESULTS.md",
        "PHASE5_SUPERVISOR_BRIEF.md",
        "PHASE5_NEXT_EXPERIMENTS.md",
    ]
    for name in names:
        assert (ROOT / "docs" / name).is_file(), name


def test_phase5_preserves_intervention_warning() -> None:
    for name in ("PHASE5_INTERVENTION_EVIDENCE.md", "PHASE5_RESULTS.md"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert WARNING in text
    app = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert WARNING in app


def test_phase5_reports_are_machine_readable() -> None:
    paths = [
        "reports/phase5/protocol_comparison/protocol_metrics.csv",
        "reports/phase5/online_decoding/online_metrics.csv",
        "reports/phase5/online_decoding/online_deltas.csv",
        "reports/phase5/mechanisms/mechanism_channel_tests.csv",
        "reports/phase5/interventions/intervention_evidence.csv",
    ]
    for rel in paths:
        assert (ROOT / rel).is_file(), rel
