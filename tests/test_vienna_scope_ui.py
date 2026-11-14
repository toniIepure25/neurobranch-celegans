"""AppTest contract for the Denis scope-capture page."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.requires_data

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "neurobranch" / "bundlenet_sim" / "app.py"


def _open_scope() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.radio[0].set_value("Scope & decisions")
    at.run()
    return at


def test_scope_page_starts_unanswered_and_unconfirmed() -> None:
    at = _open_scope()
    assert not at.exception, at.exception
    assert at.selectbox(key="scope_primary_object").value is None
    assert at.selectbox(key="scope_primary_endpoint").value is None
    assert at.multiselect(key="scope_available_data").value == []
    assert at.selectbox(key="scope_success_criterion").value is None
    assert at.selectbox(key="scope_target_mode").value is None
    assert "G0 open" in " ".join(str(item.value) for item in at.warning)
    assert len(at.get("file_uploader")) == 2
    page_text = " ".join(str(item.value) for item in at.markdown)
    assert "decision-rail" in page_text
    assert "contract-kicker" in page_text
    assert "G0 is open" in page_text
    labels = [str(button.label) for button in at.get("download_button")]
    assert not any("scope transcription" in label for label in labels)
    assert not any("decision contract" in label for label in labels)


def test_scope_page_exports_only_after_complete_provenance() -> None:
    at = _open_scope()
    at.selectbox(key="scope_primary_object").set_value(
        "A causal test of a named-neuron or circuit perturbation"
    )
    at.selectbox(key="scope_primary_endpoint").set_value("Binary reversal onset/offset")
    at.multiselect(key="scope_available_data").set_value(
        ["A randomized optogenetic/ablation intervention with matched sham"]
    )
    at.selectbox(key="scope_success_criterion").set_value("A rigorous negative result")
    at.selectbox(key="scope_target_mode").set_value("Not in scope for this project")
    at.text_area(key="scope_stop_rule").set_value("Stop after two independent batches fail.")
    at.text_area(key="scope_budget_deadline").set_value("6 ECTS, 2027-06-30.")
    at.text_input(key="scope_recorded_by").set_value("Project owner")
    at.text_input(key="scope_recorded_on").set_value("2026-11-20")
    at.selectbox(key="scope_answer_source").set_value("Written email")
    at.run()

    assert not at.exception, at.exception
    assert "Recorded, not signed" in " ".join(str(item.value) for item in at.success)
    warnings = " ".join(str(item.value) for item in at.warning)
    assert "Claim-readiness advisories" in warnings
    page_text = " ".join(str(item.value) for item in at.markdown)
    assert "causal_without_independent_labels" in page_text
    labels = [str(button.label) for button in at.get("download_button")]
    assert any("decision contract" in label for label in labels)
    assert any("scope transcription (JSON)" in label for label in labels)
    assert any("scope transcription (Markdown)" in label for label in labels)
    page_text = " ".join(str(item.value) for item in at.markdown).lower()
    assert "denis confirmed" not in page_text
    assert "exactly what denis wants" not in page_text
