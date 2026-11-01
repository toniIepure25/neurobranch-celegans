"""Safety-label contract for the context-aware dashboard view."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_context_view_keeps_required_safety_labels():
    text = (ROOT / "app" / "context_ui.py").read_text(encoding="utf-8")
    for phrase in (
        "CONTEXT-AWARE NEURAL SIMULATOR",
        "NOT CAUSALLY VALIDATED",
        "NOT_EVALUABLE",
        "measured",
        "Hypothetical oxygen",
        "model uncertainty",
        "not a bifurcation",
    ):
        assert phrase in text


def test_missing_reports_are_not_silently_substituted():
    text = (ROOT / "app" / "context_ui.py").read_text(encoding="utf-8")
    assert 'if not path.exists():' in text
    assert 'return pd.DataFrame()' in text
    assert "report not generated yet" in text


def test_context_view_is_registered_without_removing_existing_tabs():
    text = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "CONTEXT-AWARE NEURAL SIMULATOR" in text
    assert "Core Dynamics Simulator" in text
    assert "BoundedBench" in text
    assert "render_context(" in text
