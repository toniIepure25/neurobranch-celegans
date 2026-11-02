from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
KATO_PATH = ROOT / "data" / "raw" / "kato2015" / "WT_NoStim.mat"

pytestmark = pytest.mark.skipif(not KATO_PATH.exists(), reason="Kato 2015 dataset not present")


def test_app_retains_phase2_tabs_and_adds_prospective_tab():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    labels = [tab.label for tab in at.tabs]
    assert labels == [
        "Overview",
        "3-D latent",
        "Branching",
        "Virtual perturbation",
        "Optogenetic (Randi)",
        "Reproduction",
        "Audit",
        "Prospective Behavior",
        "Turn Direction & Neural Sequences",
        "SAAV Mechanistic Validation",
        "Causal Response Operator",
        "Active Probing",
        "Adaptive Twin",
        "BoundedBench",
        "Core Dynamics Simulator",
        "CONTEXT-AWARE NEURAL SIMULATOR",
        "Overview",
        "Latent model",
        "Oxygen context",
        "Branching",
        "Perturbations",
        "GENERATIVE FAILURE DIAGNOSTICS",
        "Forecastability",
        "Observation model",
        "Identifiability",
        "Oxygen response",
        "Perturbation",
        "ADAPTIVE DYNAMICS & STABILITY",
        "Overview",
        "Nonstationarity",
        "Calcium stability",
        "Adaptive models",
        "Perturbation",
    ]


def test_phase3_docs_preserve_causal_caveat():
    text = (ROOT / "docs" / "PHASE3_INTERVENTION_LIMITATIONS.md").read_text(encoding="utf-8")
    assert "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED." in text
    assert "No neuron-specific control claim is supported." in text
