from __future__ import annotations

import numpy as np
import pytest

from neurobranch.bundlenet_sim.behavior_metrics import (
    binary_metrics,
    block_bootstrap_ci,
    event_intervals,
    event_level_metrics,
    event_vector,
    lag_auc,
    persistence_labels,
    select_lag_loo,
    shift_score,
)


def _raw(n: int, intervals: list[list[int]]) -> dict[str, object]:
    return {"trace_array": [[0.0] * n], "reversal_events": intervals}


def test_event_vector_aligns_trace_offset() -> None:
    raw = _raw(30, [[7, 10], [20, 22]])
    events = event_vector(raw, 20, offset=5)
    assert events.sum() == 5
    assert np.flatnonzero(events).tolist() == [2, 3, 4, 15, 16]
    closed = event_vector(raw, 20, offset=5, interval="closed")
    assert closed.sum() == 7


def test_event_intervals_are_clipped_to_window() -> None:
    raw = _raw(30, [[2, 10], [18, 25]])
    intervals = event_intervals(raw, 20, offset=5)
    assert intervals == [(0, 5), (13, 20)]


def test_binary_metrics_known_confusion() -> None:
    events = np.array([False, False, True, True])
    labels = np.array([1, 1, -1, -1], dtype=np.int64)
    score = np.array([0.0, 0.0, 2.0, 2.0])
    metrics = binary_metrics(events, labels, score)
    assert metrics["auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["contingency"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}


def test_binary_metrics_reports_undefined_auc() -> None:
    metrics = binary_metrics(np.zeros(4, dtype=bool), np.ones(4), np.arange(4.0))
    assert metrics["auc"] is None
    assert metrics["accuracy"] == 1.0


def test_bootstrap_is_deterministic_and_reports_ci() -> None:
    rng = np.random.default_rng(7)
    events = rng.random(600) < 0.3
    score = events.astype(float) + rng.normal(0, 0.7, size=600)
    first = block_bootstrap_ci(events, np.where(events, -1, 1), score, n_resamples=200)
    second = block_bootstrap_ci(events, np.where(events, -1, 1), score, n_resamples=200)
    assert first == second
    assert first["auc_ci"] is not None
    assert first["auc_ci"][0] <= first["auc_ci"][1]
    assert first["n_valid_auc"] >= 150


def test_shift_score_and_lag_auc_recover_a_known_delay() -> None:
    events = np.zeros(300, dtype=bool)
    for start in (20, 80, 140, 200, 260):
        events[start : start + 8] = True
    # score[t] = events[t-3], so event[t] aligns with score[t+3].
    score = np.r_[np.zeros(3), events[:-3].astype(float)]
    assert lag_auc(events, score, 0) < lag_auc(events, score, 3)
    assert lag_auc(events, score, 3) > 0.99
    assert shift_score(score, 3)[0] == score[3]


def test_loo_selects_shift_on_training_recordings_only() -> None:
    records: dict[str, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(11)
    for recording_id in ("a", "b", "c"):
        events = np.zeros(500, dtype=bool)
        for start in range(20, 480, 40):
            events[start : start + 10] = True
        score = np.r_[np.zeros(4), events[:-4].astype(float)]
        score = score + rng.normal(0, 0.15, size=len(score))
        records[recording_id] = {"events": events, "score": score}
    folds = select_lag_loo(records, shifts=range(-6, 7))
    assert len(folds) == 3
    assert all(fold["held_out"] != "" for fold in folds)
    assert {fold["selected_shift_frames"] for fold in folds} == {4}
    assert all(fold["held_out_retrospective_auc"] > fold["held_out_lag0_auc"] for fold in folds)
    assert all(fold["held_out_prospective_auc"] < fold["held_out_retrospective_auc"] for fold in folds)


def test_persistence_uses_previous_frame() -> None:
    events = np.array([False, True, True, False])
    labels = persistence_labels(events)
    assert labels.tolist() == [1, 1, -1, -1]


def test_event_level_detects_latency_and_false_alarms() -> None:
    labels = np.ones(30, dtype=np.int64)
    labels[5] = -1
    labels[20] = -1
    metrics = event_level_metrics(
        events=np.zeros(30, dtype=bool),
        labels=labels,
        intervals=[(2, 6)],
        dt=0.01,
        tolerance_frames=2,
    )
    assert metrics["n_detected"] == 1
    assert metrics["median_detection_latency_s"] == pytest.approx(0.03)
    assert metrics["n_false_alarm_runs"] == 1
    assert metrics["event_recall"] == 1.0
