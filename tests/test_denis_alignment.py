"""Contract for the Denis public-source audit and stale-claim cleanup."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_audit_exists_and_is_dated() -> None:
    path = DOCS / "DENIS_PUBLIC_SOURCE_AUDIT.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    flat = " ".join(text.lower().split())
    assert "2026-11-13" in text
    assert "https://ni.cs.univie.ac.at/about-us/team/" in text
    assert "https://ucrisportal.univie.ac.at/en/persons/denis-kovacevic/" in text
    assert "no public denis-specific research statement" in flat
    assert "not proof that no publication exists" in flat
    assert "do not infer a requirement" in flat
    assert "bundle-net" in flat
    assert "nc-mcm" in flat


def test_stale_profile_url_is_not_used_outside_the_audit() -> None:
    stale = "about-us/team/person/unitid/81/denis-kovacevic"
    assert stale not in (DOCS / "LAB_ALIGNMENT.md").read_text(encoding="utf-8")
    assert stale not in (ROOT / "src" / "neurobranch" / "bundlenet_sim" / "app.py").read_text(
        encoding="utf-8"
    )
    assert stale in (DOCS / "DENIS_PUBLIC_SOURCE_AUDIT.md").read_text(encoding="utf-8")


def test_ui_has_no_stale_contract_count() -> None:
    app = (ROOT / "src" / "neurobranch" / "bundlenet_sim" / "app.py").read_text(encoding="utf-8")
    assert "40 contract tests" not in app
    assert '"Contract tests"' not in app


def test_canonical_brief_links_audit_and_keeps_the_unsigned_gate() -> None:
    text = (DOCS / "DENIS_REQUIREMENTS_BRIEF.md").read_text(encoding="utf-8")
    assert "DENIS_PUBLIC_SOURCE_AUDIT.md" in text
    assert "unconfirmed_transcription" in text
    assert "A transcription is not a signature" in text
    assert "Confirmed by" in text


def test_audit_is_in_the_docs_nav() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "DENIS_PUBLIC_SOURCE_AUDIT.md" in nav
