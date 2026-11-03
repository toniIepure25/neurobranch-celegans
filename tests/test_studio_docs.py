"""Documentation contract for the Scientific Studio handoff."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

REQUIRED = [
    "STUDIO_ARCHITECTURE.md",
    "STUDIO_DESIGN_SYSTEM.md",
    "STUDIO_3D_SCIENTIFIC_MAPPING.md",
    "STUDIO_USER_GUIDE.md",
    "STUDIO_SCIENTIFIC_EVIDENCE.md",
    "STUDIO_DOCUMENTATION_ERRATA.md",
    "STUDIO_REPRODUCIBILITY.md",
    "STUDIO_SUPERVISOR_DEMO.md",
    "STUDIO_TESTING_AND_QA.md",
    "STUDIO_FINAL_NUMERICAL_AUDIT.md",
    "STUDIO_FINAL_PCA_AUDIT.md",
    "STUDIO_FINAL_SCIENTIFIC_QA.md",
    "STUDIO_FINAL_VISUAL_QA.md",
    "STUDIO_FINAL_USER_GUIDE.md",
    "STUDIO_FINAL_SUPERVISOR_CHECKLIST.md",
]


@pytest.mark.parametrize("name", REQUIRED)
def test_required_studio_document_exists(name: str) -> None:
    path = DOCS / name
    assert path.is_file(), name
    assert path.stat().st_size > 1500, name


def test_current_status_points_to_studio() -> None:
    text = (DOCS / "CURRENT_STATUS.md").read_text(encoding="utf-8")
    assert "STUDIO_USER_GUIDE.md" in text
    assert "+0.2644" in text
    assert "NOT_READY_FOR_CONFIRMATORY_OR_BIOLOGICAL_CLAIMS" in text


def test_active_data_provenance_has_verified_size() -> None:
    text = (DOCS / "DATA_PROVENANCE.md").read_text(encoding="utf-8")
    assert "56,390,975" in text
    assert "128,511,040" not in text


def test_mkdocs_nav_contains_all_studio_docs() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for name in REQUIRED:
        assert name in nav, name


def test_studio_relative_links_resolve() -> None:
    missing: list[str] = []
    for path in DOCS.glob("STUDIO_*.md"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
            target = match.group(1).split("#")[0]
            if not target or target.startswith(("http", "mailto:", "#")):
                continue
            if not (path.parent / target).resolve().exists():
                missing.append(f"{path.name}: {target}")
    assert not missing, missing


def test_errata_covers_all_seventeen() -> None:
    text = (DOCS / "STUDIO_DOCUMENTATION_ERRATA.md").read_text(encoding="utf-8")
    for index in range(1, 18):
        assert f"| X{index:02d} |" in text
