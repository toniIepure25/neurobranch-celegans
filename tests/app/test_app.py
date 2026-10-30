"""App smoke tests for the BoundedBench provenance surface."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")


def test_app_builds_all_tabs_without_exception() -> None:
    app = AppTest.from_file(APP)
    app.run(timeout=90)
    assert not app.exception
    assert len(app.tabs) == 14


def test_boundedbench_shows_provenance() -> None:
    app = AppTest.from_file(APP)
    app.run(timeout=90)
    captions = " ".join(item.value for item in app.caption)
    assert "Card 2.0" in captions
    assert "manifest" in captions.lower()
    warnings = " ".join(item.value for item in app.warning)
    assert "hierarchical interval is not calibrated" in warnings.lower()
    assert "0.560" in warnings
