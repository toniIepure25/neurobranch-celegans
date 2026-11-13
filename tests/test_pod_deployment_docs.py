"""Contract for the `train` pod deployment runbook.

The deployment is infrastructure, not evidence. This test keeps the runbook
pointed at the verified pod, paths and ports so a future edit cannot silently
document a different target.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_train_deployment_document_is_substantive() -> None:
    path = DOCS / "POD_DEPLOYMENT_TRAIN.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert len(text) > 3000
    for required in [
        "train-01-0-0",
        "runai-romania-dev",
        "/work/neurobranch-celegans",
        "8502",
        "8504",
        "kubectl port-forward",
        "PYTHONPATH",
        "Atanas_data",
        "sha256sum",
        "streamlit",
    ]:
        assert required in text, required


def test_train_deployment_keeps_the_claim_boundary() -> None:
    text = (DOCS / "POD_DEPLOYMENT_TRAIN.md").read_text(encoding="utf-8")
    assert "not a scientific result" in text
    assert "does not add data" in text
    assert "local-only" in text


def test_train_deployment_is_linked_from_readme_and_nav() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "POD_DEPLOYMENT_TRAIN.md" in readme
    assert "POD_DEPLOYMENT_TRAIN.md" in nav
