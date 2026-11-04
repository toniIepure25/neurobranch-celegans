"""Streamlit app smoke tests using the supported test harness."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.requires_data

APP = Path(__file__).resolve().parents[1] / "app" / "neurobranch_studio.py"


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120)


def test_overview_renders_without_exception() -> None:
    at = _app().run()
    assert not at.exception, at.exception
    assert "NeuroBranch Scientific Studio" in " ".join(
        str(m.value) for m in at.markdown
    ) or at.title


def test_dynamics_route_renders_without_exception() -> None:
    at = _app().run()
    at.session_state["studio_section"] = "dynamics"
    at.run()
    assert not at.exception, at.exception
    assert any("state space" in str(m.value).lower() for m in at.markdown)


def test_models_route_renders_without_exception() -> None:
    at = _app().run()
    at.session_state["studio_section"] = "models"
    at.run()
    assert not at.exception, at.exception


@pytest.mark.parametrize(
    "section", ["activity", "evidence", "perturbations", "notebook"]
)
def test_remaining_routes_render_without_exception(section: str) -> None:
    at = _app().run()
    at.session_state["studio_section"] = section
    at.run()
    assert not at.exception, at.exception


def test_light_mode_renders_without_exception() -> None:
    at = _app().run()
    at.session_state["studio_light"] = True
    at.run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("index", range(10))
def test_presentation_slide_renders_without_exception(index: int) -> None:
    at = _app().run()
    at.session_state["studio_present"] = True
    at.session_state["studio_present_index"] = index
    at.run()
    assert not at.exception, at.exception
    if index == 0:
        assert any("bounded result" in str(m.value).lower() for m in at.markdown)
    if index == 1:
        assert any("recordings" in str(m.value).lower() for m in at.markdown)


def test_slide_two_metrics_and_cohort_wording() -> None:
    at = _app().run()
    at.session_state["studio_present"] = True
    at.session_state["studio_present_index"] = 1
    at.run()
    assert not at.exception, at.exception
    labels = [str(metric.label) for metric in at.metric]
    assert labels == ["Neural recordings", "Frames", "Included predictors"]
    text = " ".join(str(m.value) for m in at.markdown).lower()
    assert "one experimental cohort" in text
    assert "distinct preparations" not in text
    assert "one preparation" not in text
