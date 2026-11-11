"""Sixth-worm manifest gate tests."""

from __future__ import annotations

import json
from pathlib import Path

from neurobranch.data.validation import validate_new_worm_manifest

SHA = "b" * 64


def _manifest(path: Path, **overrides: object) -> Path:
    payload = {
        "cohort_id": "sixth-worm",
        "source": "lab",
        "license": "CC-BY-4.0",
        "sha256": SHA,
        "unit": "animal",
        "n_units": 1,
        "timebase_s": 0.5,
        "files": ["embedding.npz"],
        "behaviors": ["forward", "reverse"],
        "intervention": False,
        "allowed_uses": ["external_only"],
        "owner": "Denis",
        "notes": "",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_sixth_worm_manifest_passes(tmp_path: Path) -> None:
    (tmp_path / "embedding.npz").write_bytes(b"fake")
    path = _manifest(tmp_path / "manifest.json")
    report = validate_new_worm_manifest(path, data_root=tmp_path)
    assert report["passed"] is True
    assert report["cohort_id"] == "sixth-worm"


def test_missing_file_fails(tmp_path: Path) -> None:
    path = _manifest(tmp_path / "manifest.json")
    report = validate_new_worm_manifest(path, data_root=tmp_path)
    assert report["passed"] is False
    assert any("does not exist" in error for error in report["errors"])


def test_training_use_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "embedding.npz").write_bytes(b"fake")
    path = _manifest(tmp_path / "manifest.json", allowed_uses=["train"])
    report = validate_new_worm_manifest(path, data_root=tmp_path)
    assert report["passed"] is False
    assert any("external" in error or "train" in error for error in report["errors"])


def test_wrong_unit_fails(tmp_path: Path) -> None:
    (tmp_path / "embedding.npz").write_bytes(b"fake")
    path = _manifest(tmp_path / "manifest.json", unit="recording")
    report = validate_new_worm_manifest(path, data_root=tmp_path)
    assert report["passed"] is False
    assert any("unit" in error for error in report["errors"])


def test_empty_behaviors_fails(tmp_path: Path) -> None:
    (tmp_path / "embedding.npz").write_bytes(b"fake")
    path = _manifest(tmp_path / "manifest.json", behaviors=[])
    report = validate_new_worm_manifest(path, data_root=tmp_path)
    assert report["passed"] is False
    assert any("behavio" in error for error in report["errors"])
