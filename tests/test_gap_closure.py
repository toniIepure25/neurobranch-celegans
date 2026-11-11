"""Contract for the gap-closure handoff and UI section."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

REQUIRED = [
    "DENIS_REQUIREMENTS_BRIEF.md",
    "GAP_CLOSURE_MATRIX.md",
    "DATA_REQUEST_PACKAGE.md",
    "ANNOTATION_PROTOCOL.md",
    "INTERVENTION_POWER_PLAN.md",
    "PUBLIC_DATA_GAP_MEMO.md",
]


def test_gap_documents_exist_and_are_substantive() -> None:
    for name in REQUIRED:
        path = DOCS / name
        assert path.is_file(), name
        assert path.stat().st_size > 1500, name


def test_gap_matrix_keeps_the_verified_coverage_limits() -> None:
    text = (DOCS / "GAP_CLOSURE_MATRIX.md").read_text(encoding="utf-8")
    assert "0/27" in text
    assert "0/30" in text
    assert "blocked on data" in text.lower()
    assert "digital twin" in text.lower()


def test_data_request_names_the_minimal_artifacts() -> None:
    text = (DOCS / "DATA_REQUEST_PACKAGE.md").read_text(encoding="utf-8")
    assert "bundlenet_consistent_embedding_worm_5.npz" in text
    assert "clock_offset_ms" in text
    assert "Fleiss" in text


def test_brief_does_not_claim_denis_confirmed_it() -> None:
    text = (DOCS / "DENIS_REQUIREMENTS_BRIEF.md").read_text(encoding="utf-8").lower()
    assert "draft" in text
    assert "not confirmed" in text
    assert "confirmed by" in text


def test_lab_tab_exposes_gap_closure_without_new_claim() -> None:
    app = (ROOT / "src" / "neurobranch" / "bundlenet_sim" / "app.py").read_text(encoding="utf-8")
    assert "def gap_closure_section" in app
    assert "Gap closure" in app
    assert "blocked on data" in app
    assert "not claimed — L0 only" in app


def test_gap_nav_contains_all_new_documents() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for name in REQUIRED:
        assert name in nav, name


@pytest.mark.requires_data
def test_gap_closure_page_renders_with_documents() -> None:
    from streamlit.testing.v1 import AppTest

    app = ROOT / "src" / "neurobranch" / "bundlenet_sim" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    at.radio[0].set_value("Evidence & lab")
    at.run()
    assert not at.exception, at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    assert "Gap closure" in text
    labels = [str(button.label) for button in at.get("download_button")]
    assert any("Denis brief" in label for label in labels)
    assert any("Gap matrix" in label for label in labels)
    gap_tables = [
        frame
        for frame in at.dataframe
        if "gate" in {str(column).lower() for column in frame.value.columns}
    ]
    assert gap_tables
    assert "blocked on data" in " ".join(gap_tables[0].value["status"].astype(str)).lower()
