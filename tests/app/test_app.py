"""App smoke tests for the BoundedBench provenance surface."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "app" / "streamlit_app.py")
KATO_PATH = ROOT / "data" / "raw" / "kato2015" / "WT_NoStim.mat"

pytestmark = pytest.mark.skipif(not KATO_PATH.exists(), reason="Kato 2015 dataset not present")


def test_app_builds_all_tabs_without_exception() -> None:
    app = AppTest.from_file(APP)
    app.run(timeout=90)
    assert not app.exception
    assert len(app.tabs) == 33
    labels = [tab.label for tab in app.tabs]
    assert "CONTEXT-AWARE NEURAL SIMULATOR" in labels
    assert "GENERATIVE FAILURE DIAGNOSTICS" in labels
    assert "ADAPTIVE DYNAMICS & STABILITY" in labels
    assert "Latent model" in labels
    assert "Oxygen context" in labels


def test_boundedbench_shows_provenance() -> None:
    app = AppTest.from_file(APP)
    app.run(timeout=90)
    captions = " ".join(item.value for item in app.caption)
    assert "Card 2.0" in captions
    assert "manifest" in captions.lower()
    warnings = " ".join(item.value for item in app.warning)
    assert "hierarchical interval is not calibrated" in warnings.lower()
    assert "R_22" in warnings
    assert "counterfactual" in warnings.lower()
