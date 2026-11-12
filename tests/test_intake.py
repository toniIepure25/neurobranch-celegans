"""Tests for the in-session incoming-data validation helpers."""

from __future__ import annotations

import pytest

from neurobranch.bundlenet_sim.intake import (
    MAX_UPLOAD_BYTES,
    validate_annotation_uploads,
    validate_trial_upload,
)

ANNOTATION_HEADER = "recording_id,frame_start,frame_end,behavior_label,annotator_id,confidence,notes\n"


def _annotation_rows(annotator: str) -> str:
    return (
        f"r1,0,10,forward,{annotator},1,\n"
        f"r1,10,20,reverse,{annotator},1,\n"
        f"r1,20,30,forward,{annotator},1,\n"
    )


ANNOTATION_CSV = ANNOTATION_HEADER + _annotation_rows("a1") + _annotation_rows("a2")

TRIAL_CSV = (
    "animal_id,session_id,arm,target,sham_type,intervention_start_frame,"
    "intervention_end_frame,fps,behavior_label_path,neural_trace_path,clock_offset_ms,"
    "annotator_id\n"
    + "".join(
        f"t{i},s{i},target,AVB,,{i * 20},{i * 20 + 5},10,behavior.csv,neural.npy,0,blind\n"
        for i in range(6)
    )
    + "".join(
        f"s{i},s{i},sham,,same-neuron-off,{i * 20 + 10},{i * 20 + 15},10,"
        "behavior.csv,neural.npy,0,blind\n"
        for i in range(6)
    )
)


def test_annotation_upload_passes_gate() -> None:
    report = validate_annotation_uploads([("a1.csv", ANNOTATION_CSV.encode())])
    assert report["passed"] is True
    assert report["source"] == "a1.csv"
    assert report["macro_kappa"] == pytest.approx(1.0)


def test_multiple_annotation_files_are_compared() -> None:
    report = validate_annotation_uploads(
        [
            ("a1.csv", (ANNOTATION_HEADER + _annotation_rows("a1")).encode()),
            ("a2.csv", (ANNOTATION_HEADER + _annotation_rows("a2")).encode()),
        ]
    )
    assert report["passed"] is True
    assert report["n_annotators"] == 2
    assert report["source"] == "a1.csv, a2.csv"


def test_trial_upload_passes_gate() -> None:
    report = validate_trial_upload("trials.csv", TRIAL_CSV.encode())
    assert report["passed"] is True
    assert report["source"] == "trials.csv"
    assert report["animal_counts"] == {"sham": 6, "target": 6}


def test_upload_rejects_empty_wrong_extension_and_oversize() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_annotation_uploads([])
    with pytest.raises(ValueError, match="CSV"):
        validate_annotation_uploads([("a1.txt", b"x")])
    with pytest.raises(ValueError, match="too large"):
        validate_trial_upload("trials.csv", b"x" * (MAX_UPLOAD_BYTES + 1))
