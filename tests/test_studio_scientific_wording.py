"""Regression tests for supervisor-facing scientific wording."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.requires_data

APP = Path(__file__).resolve().parents[1] / "app" / "neurobranch_studio.py"


def _text(at: AppTest) -> str:
    parts = [str(m.value) for m in at.markdown]
    parts.extend(str(c.value) for c in at.caption)
    return " ".join(parts).lower()


def test_overview_uses_reported_not_prespecified_contrast() -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "reported contrast" in text
    assert "prespecified contrast" not in text
    assert "sign convention was clarified" in text
    assert "one experimental cohort" in text
    assert "laboratory / preparation" not in text
    assert "distinct acquisition preparations" not in text
    assert "five_distinct_preparations" not in text


@pytest.mark.parametrize("section", ["evidence", "models"])
def test_active_sections_do_not_repeat_old_cohort_language(section: str) -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.session_state["studio_section"] = section
    at.run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "distinct preparations" not in text
    assert "prespecified contrast" not in text


def test_footer_scopes_test_tail_reuse_to_this_release() -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "no additional test-tail model optimization" in text
    assert "earlier research phases reused the evaluation tails" in text
    assert "no test-tail performance was optimized" not in text
