"""Smoke tests for the focused supervisor demo."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION \u2014 NOT CAUSALLY VALIDATED."

SECTIONS = {
    "Start here": "What this is",
    "Timeline": "Chronological account",
    "Central result": "The central contrast",
    "Real trajectory": "A real observed continuation",
    "Mean vs stochastic": "Conditional mean vs stochastic rollout",
    "Calibration": "The mean can be useful",
    "Perturbation boundary": "A mathematical model edit",
    "NEURAL INTERACTION & SENSITIVITY EXPLORER": "NEURAL INTERACTION & SENSITIVITY EXPLORER",
    "Decision and next": "Do not run another architecture search",
}


def _run() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app" / "handoff_demo.py"), default_timeout=90)
    at.run()
    return at


def test_default_section_renders_without_exception() -> None:
    at = _run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Start here"
    assert any("What this is" in item.value for item in at.subheader)
    labels = [metric.label for metric in at.metric]
    assert "Sessions" in labels
    assert "H20 landmarks" in labels


def test_every_presentation_section_renders() -> None:
    at = _run()
    for section, expected in SECTIONS.items():
        at.sidebar.radio[0].set_value(section).run()
        assert not at.exception, f"{section}: {[e.value for e in at.exception]}"
        assert any(expected in item.value for item in at.subheader), section


def test_perturbation_section_shows_exact_warning() -> None:
    at = _run()
    at.sidebar.radio[0].set_value("Perturbation boundary").run()
    assert not at.exception
    errors = [item.value for item in at.error]
    assert any(WARNING in value for value in errors)
    metric_values = [item.value for item in at.metric]
    assert "NOT_EVALUABLE" in metric_values
    assert any("not optogenetics" in item.value for item in at.markdown)


def test_decision_section_keeps_the_strongest_allowed_sentence() -> None:
    at = _run()
    at.sidebar.radio[0].set_value("Decision and next").run()
    assert not at.exception
    infos = [item.value for item in at.info]
    assert any("fixed 10-D PCA" in value and "unresolved" in value for value in infos)


def test_interaction_explorer_is_optional_and_shows_warning() -> None:
    at = _run()
    at.sidebar.radio[0].set_value("NEURAL INTERACTION & SENSITIVITY EXPLORER").run()
    assert not at.exception
    assert any("NEURAL INTERACTION & SENSITIVITY EXPLORER" in item.value for item in at.subheader)
    if any("optional interaction artifacts are absent" in item.value for item in at.info):
        return
    errors = [item.value for item in at.error]
    assert any(WARNING in value for value in errors)
    assert "NOT_EVALUABLE" in " ".join(item.value for item in at.markdown)
