"""Intervention trial validator tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from neurobranch.data.validation import validate_intervention_trials

FIELDS = [
    "animal_id",
    "session_id",
    "arm",
    "target",
    "sham_type",
    "intervention_start_frame",
    "intervention_end_frame",
    "behavior_label_path",
    "neural_trace_path",
    "fps",
    "clock_offset_ms",
    "annotator_id",
    "notes",
]


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _row(animal: str, arm: str, start: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "animal_id": animal,
        "session_id": "s1",
        "arm": arm,
        "target": "AVAL" if arm == "target" else "",
        "sham_type": "light-only" if arm == "sham" else "",
        "intervention_start_frame": start,
        "intervention_end_frame": start + 10,
        "behavior_label_path": "behaviour.csv",
        "neural_trace_path": "neural.npy",
        "fps": 10.0,
        "clock_offset_ms": 0.0,
        "annotator_id": "a1",
        "notes": "",
    }
    row.update(overrides)
    return row


def test_balanced_trial_table_passes(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    rows = [_row(f"a{i}", "target", i * 20) for i in range(3)]
    rows += [_row(f"a{i}", "sham", i * 20 + 5) for i in range(3)]
    _write(path, rows)
    report = validate_intervention_trials(path, min_per_arm=3)
    assert report["passed"] is True
    assert report["n_animals"] == 3
    assert report["arm_counts"] == {"sham": 3, "target": 3}


def test_missing_clock_fails(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    _write(
        path,
        [
            _row("a1", "target", 0, clock_offset_ms=""),
            _row("a1", "sham", 20),
        ],
    )
    report = validate_intervention_trials(path, min_per_arm=1)
    assert report["passed"] is False
    assert any("clock_offset_ms" in error for error in report["errors"])


def test_duplicate_trial_fails(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    _write(
        path,
        [
            _row("a1", "target", 0),
            _row("a1", "target", 0),
            _row("a1", "sham", 20),
        ],
    )
    report = validate_intervention_trials(path, min_per_arm=1)
    assert report["passed"] is False
    assert any("duplicate" in error for error in report["errors"])


def test_imbalanced_table_fails_gate(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    rows = [_row(f"a{i}", "target", i * 20) for i in range(3)]
    rows += [_row(f"a{i}", "sham", i * 20 + 5) for i in range(1)]
    _write(path, rows)
    report = validate_intervention_trials(path, min_per_arm=3)
    assert report["passed"] is False
    assert any("arm" in error for error in report["errors"])


def test_minimum_counts_animals_not_trials(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    rows = [_row("a1", "target", i * 20) for i in range(6)]
    rows += [_row("a1", "sham", i * 20 + 5) for i in range(6)]
    _write(path, rows)
    report = validate_intervention_trials(path, min_per_arm=6)
    assert report["passed"] is False
    assert report["animal_counts"] == {"sham": 1, "target": 1}
    assert any("animals" in error for error in report["errors"])


def test_unknown_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("animal_id,arm\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        validate_intervention_trials(path)
