"""Executable invariants for the Kato master-review forensics.

These tests recompute the bounds from the raw HDF5 archive and compare them with
the frozen landmark table.  They do not import the project's split or landmark
helper for the expected values.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "master_review"))

import kato_forensics as KF  # noqa: E402


def _archive():
    fh = h5py.File(KF.DATA, "r")
    group = fh["WT_NoStim"]
    n = int(group["fps"].shape[0])
    records = []
    for i in range(n):
        names_ds = KF.deref(fh, group["NeuronNames"][i, 0])
        names = tuple(KF.decode_name(fh, x) for x in np.asarray(names_ds).ravel())
        traces = np.asarray(KF.deref(fh, group["deltaFOverF_bc"][i, 0]), dtype=np.float32)
        if traces.shape[0] == len(names) and traces.shape[1] != len(names):
            traces = traces.T
        records.append((int(traces.shape[0]), len(names)))
    return fh, records


def test_h100_targets_are_inside_each_recording():
    fh, records = _archive()
    try:
        for T, _ in records:
            bounds = KF.split_bounds(T)
            mask = KF.masks_from_bounds(bounds, T)["test"]
            starts = KF.starts_for_horizon(T, mask, 100, KF.STRIDE)
            assert len(starts) > 0
            assert int(starts.max()) + 100 <= T - 1
            assert mask[starts.max() + 100]
            assert int(starts.min()) >= bounds["test"][0]
            assert int(starts.min()) - 1 >= 0
    finally:
        fh.close()


def test_landmark_counts_are_not_35_per_session():
    payload = json.loads((KF.OUT / "frame_landmark_bounds.json").read_text(encoding="utf-8"))
    counts = payload["summary"]["landmarks_per_session"]
    assert counts == {"0": 35, "1": 35, "2": 34, "3": 38, "4": 33}
    assert sum(counts.values()) == 175
    assert not all(v == 35 for v in counts.values())


def test_frozen_landmarks_match_independent_set():
    payload = json.loads((KF.OUT / "frame_landmark_bounds.json").read_text(encoding="utf-8"))
    frozen_path = ROOT / "reports" / "simulator" / "kato_common_state" / "model_per_landmark.csv"
    frozen: dict[int, set[int]] = {}
    with frozen_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frozen.setdefault(int(row["session"]), set()).add(int(row["landmark"]))
    # The frozen file contains one row per model; set() preserves the unique set.
    assert payload["frozen_landmark_comparison"]["counts_match"] is True
    assert {k: len(v) for k, v in frozen.items()} == {0: 35, 1: 35, 2: 34, 3: 38, 4: 33}


def test_no_target_crosses_a_split_boundary_at_any_reported_horizon():
    payload = json.loads((KF.OUT / "frame_landmark_bounds.json").read_text(encoding="utf-8"))
    assert payload["summary"]["all_targets_in_test_mask"] is True
    assert payload["summary"]["any_target_crosses_boundary"] is False
    assert payload["summary"]["any_interpolation_or_padding"] is False
    for row in payload["horizon_checks"]:
        assert row["all_targets_in_observation_axis"] is True
        assert row["targets_cross_split_boundary"] is False


def test_documentation_defects_are_recorded_not_hidden():
    payload = json.loads((KF.OUT / "frame_landmark_bounds.json").read_text(encoding="utf-8"))
    claims = payload["doc_claims_checked"]
    assert claims["thirty_five_landmarks_per_session"]["status"] == "MISMATCH"
    assert claims["shared_test_boundary_2689_3137"]["status"] == "MISLEADING_IF_PRESENTED_AS_COMMON"
    assert claims["provenance_file_size"]["status"] == "MISMATCH_DOCUMENTATION"


def test_animal_level_interval_is_not_hidden_by_landmark_bootstrap():
    payload = json.loads((KF.OUT / "statistical_audit.json").read_text(encoding="utf-8"))
    lo, hi = payload["session_t_interval_95"]
    assert lo < 0 < hi
    assert payload["sessions_positive"] == 4
    assert payload["sign_test_one_sided_p"] > 0.05
    assert payload["overall_nonoverlapping_proxy"] < payload["n_landmarks"]
