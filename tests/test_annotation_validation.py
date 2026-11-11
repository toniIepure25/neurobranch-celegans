"""Annotation validator tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from neurobranch.data.validation import (
    validate_annotation_csv,
    validate_annotation_directory,
)


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    fields = [
        "recording_id",
        "frame_start",
        "frame_end",
        "behavior_label",
        "annotator_id",
        "confidence",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _rows(labels: list[str], annotator: str, recording: str = "r1") -> list[dict[str, object]]:
    return [
        {
            "recording_id": recording,
            "frame_start": index * 10,
            "frame_end": (index + 1) * 10,
            "behavior_label": label,
            "annotator_id": annotator,
            "confidence": 1.0,
            "notes": "",
        }
        for index, label in enumerate(labels)
    ]


def test_perfect_agreement_passes(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    _write(path, _rows(["forward", "reverse", "forward"], "a1") + _rows(["forward", "reverse", "forward"], "a2"))
    report = validate_annotation_csv(path)
    assert report["passed"] is True
    assert report["macro_kappa"] == pytest.approx(1.0)
    assert report["n_annotators"] == 2


def test_disagreement_fails_gate(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    _write(path, _rows(["forward", "reverse", "forward", "reverse"], "a1") + _rows(["forward", "forward", "reverse", "reverse"], "a2"))
    report = validate_annotation_csv(path, require_kappa=True)
    assert report["passed"] is False
    assert report["macro_kappa"] < 0.6


def test_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("recording_id,frame_start\nr1,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        validate_annotation_csv(path)


def test_overlapping_intervals_raise(tmp_path: Path) -> None:
    path = tmp_path / "overlap.csv"
    _write(
        path,
        [
            {"recording_id": "r1", "frame_start": 0, "frame_end": 20, "behavior_label": "forward", "annotator_id": "a1", "confidence": 1.0, "notes": ""},
            {"recording_id": "r1", "frame_start": 10, "frame_end": 30, "behavior_label": "reverse", "annotator_id": "a1", "confidence": 1.0, "notes": ""},
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_annotation_csv(path)


def test_directory_requires_two_annotators(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    _write(path, _rows(["forward"], "a1"))
    report = validate_annotation_directory(tmp_path)
    assert report["passed"] is False
    assert any("two annotators" in error for error in report["errors"])
