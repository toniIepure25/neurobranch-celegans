"""Behavioural evaluation utilities for the Tier-3 clamped replay.

The functions in this module are deliberately independent of Streamlit and of
the report builder.  They implement the definitions used by
``scripts/build_tier3_behavior_validation.py``:

* event intervals are 0-based and normally half-open;
* the primary trace window starts five raw frames after the recording start;
* a positive score means "more likely reversal";
* a positive shift compares event[t] with score[t + shift], so a positive
  selected shift means the model response follows the event (retrospective
  alignment).  A negative shift tests whether the score leads the event
  (prospective alignment).

The distinction matters: a high retrospective AUC is not evidence that the
model forecasts behaviour.  The report and UI must always show both directions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

DEFAULT_SHIFTS = tuple(range(-20, 21))
DEFAULT_BLOCK_FRAMES = 100
DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 20261111


def event_vector(
    raw: Mapping[str, object],
    n_trace: int,
    offset: int = 5,
    interval: str = "half_open",
) -> np.ndarray:
    """Map raw reversal intervals onto the trace window.

    ``interval="half_open"`` uses ``[start, end)``; ``interval="closed"`` uses
    ``[start, end]``.  The release does not document the convention, so the
    report includes a sensitivity analysis over both.
    """
    if interval not in {"half_open", "closed"}:
        raise ValueError(f"unknown interval convention: {interval}")
    trace = raw.get("trace_array")
    if not isinstance(trace, Sequence) or not trace:
        raise ValueError("recording has no trace_array")
    raw_len = len(trace[0])  # type: ignore[index]
    if offset < 0 or offset + n_trace > raw_len:
        raise ValueError(f"trace window [{offset}, {offset + n_trace}) exceeds {raw_len} frames")
    vectors = _raw_event_arrays(raw, raw_len, interval)
    return vectors[offset : offset + n_trace]


def _raw_event_arrays(
    raw: Mapping[str, object], raw_len: int, interval: str = "half_open"
) -> np.ndarray:
    events = np.zeros(raw_len, dtype=bool)
    for start, end in raw.get("reversal_events", []):  # type: ignore[union-attr]
        start, end = int(start), int(end)
        end_exclusive = end + 1 if interval == "closed" else end
        if not (0 <= start < end_exclusive <= raw_len):
            raise ValueError(f"invalid reversal interval [{start}, {end}) for {raw_len} frames")
        events[start:end_exclusive] = True
    return events


def event_intervals(
    raw: Mapping[str, object],
    n_trace: int,
    offset: int = 5,
    interval: str = "half_open",
) -> list[tuple[int, int]]:
    """Return clipped event intervals in trace coordinates."""
    trace = raw.get("trace_array")
    if not isinstance(trace, Sequence) or not trace:
        raise ValueError("recording has no trace_array")
    raw_len = len(trace[0])  # type: ignore[index]
    intervals: list[tuple[int, int]] = []
    for start, end in raw.get("reversal_events", []):  # type: ignore[union-attr]
        start, end = int(start), int(end)
        end_exclusive = end + 1 if interval == "closed" else end
        if not (0 <= start < end_exclusive <= raw_len):
            raise ValueError(f"invalid reversal interval [{start}, {end}) for {raw_len} frames")
        lo = max(0, start - offset)
        hi = min(n_trace, end_exclusive - offset)
        if hi > lo:
            intervals.append((lo, hi))
    return intervals


def shift_score(score: np.ndarray, shift: int) -> np.ndarray:
    """Align ``score[t + shift]`` to event time ``t``; pad outside with NaN."""
    score = np.asarray(score, dtype=float)
    out = np.full_like(score, np.nan)
    n = len(score)
    if shift == 0:
        return score.copy()
    if shift > 0:
        if shift < n:
            out[: n - shift] = score[shift:]
    else:
        if -shift < n:
            out[-shift:] = score[: n + shift]
    return out


def shift_labels(labels: np.ndarray, shift: int, fill: int = 0) -> np.ndarray:
    """Align ``labels[t + shift]`` to event time ``t``; pad with ``fill``."""
    labels = np.asarray(labels, dtype=np.int64)
    out = np.full_like(labels, fill)
    n = len(labels)
    if shift == 0:
        return labels.copy()
    if shift > 0:
        if shift < n:
            out[: n - shift] = labels[shift:]
    else:
        if -shift < n:
            out[-shift:] = labels[: n + shift]
    return out


def _valid_arrays(
    events: np.ndarray, labels: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    events = np.asarray(events, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)
    score = np.asarray(score, dtype=float)
    if not (len(events) == len(labels) == len(score)):
        raise ValueError("events, labels and score must have the same length")
    valid = np.isfinite(score)
    return events[valid], labels[valid], score[valid]


def binary_metrics(
    events: np.ndarray, labels: np.ndarray, score: np.ndarray
) -> dict[str, object]:
    """Frame-level binary reversal metrics.

    ``score`` must be oriented so that larger values mean reversal.  AUC is
    ``None`` when one class is absent; the reason is reported rather than
    silently returning 0.5.
    """
    events, labels, score = _valid_arrays(events, labels, score)
    n = len(events)
    if n == 0:
        return {
            "n_frames": 0,
            "auc": None,
            "accuracy": None,
            "balanced_accuracy": None,
            "reason": "no finite scores",
        }
    predicted = labels == -1
    tp = int(np.sum(predicted & events))
    fp = int(np.sum(predicted & ~events))
    fn = int(np.sum(~predicted & events))
    tn = int(np.sum(~predicted & ~events))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    balanced = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    accuracy = (tp + tn) / n
    observed = accuracy
    expected = ((tp + fp) / n) * ((tp + fn) / n) + ((fn + tn) / n) * ((fp + tn) / n)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    auc = None
    if events.any() and (~events).any():
        auc = float(roc_auc_score(events, score))
    return {
        "n_frames": n,
        "event_prevalence": float(events.mean()),
        "predicted_reverse_rate": float(predicted.mean()),
        "auc": auc,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced) if balanced is not None else None,
        "precision": float(precision) if precision is not None else None,
        "recall": float(recall) if recall is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "f1": float(f1) if f1 is not None else None,
        "cohen_kappa": float(kappa) if kappa is not None else None,
        "contingency": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def block_bootstrap_ci(
    events: np.ndarray,
    labels: np.ndarray,
    score: np.ndarray,
    block_frames: int = DEFAULT_BLOCK_FRAMES,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Block-bootstrap percentile CI for AUC and binary agreement.

    Blocks preserve local autocorrelation.  Samples with a single class are
    skipped because AUC is undefined; the report records how many were valid.
    """
    events, labels, score = _valid_arrays(events, labels, score)
    n = len(events)
    block = min(block_frames, max(1, n // 5))
    if n == 0 or block < 1:
        return {"method": "block_bootstrap", "auc_ci": None, "agreement_ci": None}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    aucs: list[float] = []
    agreements: list[float] = []
    for _ in range(n_resamples):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        index = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
        metrics = binary_metrics(events[index], labels[index], score[index])
        if metrics["auc"] is not None:
            aucs.append(float(metrics["auc"]))
        if metrics["accuracy"] is not None:
            agreements.append(float(metrics["accuracy"]))
    auc_ci = _percentile_ci(aucs)
    agreement_ci = _percentile_ci(agreements)
    return {
        "method": "block_bootstrap_percentile",
        "block_frames": int(block),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "n_frames": int(n),
        "n_valid_auc": len(aucs),
        "auc_ci": auc_ci,
        "agreement_ci": agreement_ci,
    }


def _percentile_ci(values: Iterable[float]) -> list[float] | None:
    values = np.asarray(list(values), dtype=float)
    if len(values) < 50:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def lag_auc(events: np.ndarray, score: np.ndarray, shift: int) -> float | None:
    """AUC for ``event[t]`` against ``score[t + shift]``."""
    metrics = binary_metrics(events, np.zeros(len(events), dtype=np.int64), shift_score(score, shift))
    return metrics["auc"]  # type: ignore[return-value]


def lag_curve(
    events: np.ndarray, score: np.ndarray, shifts: Iterable[int] = DEFAULT_SHIFTS
) -> list[dict[str, object]]:
    """AUC as a function of the score shift (positive = model lags event)."""
    return [
        {"shift_frames": int(shift), "auc": lag_auc(events, score, int(shift))}
        for shift in shifts
    ]


def select_lag_loo(
    records: Mapping[str, Mapping[str, np.ndarray]],
    shifts: Iterable[int] = DEFAULT_SHIFTS,
) -> list[dict[str, object]]:
    """Select a score shift on the other recordings and score the held-out one.

    A positive selected shift means the model response follows the event.  The
    held-out recording is never used to choose the shift.  Ties are broken by
    smaller absolute shift, then by the more negative shift.
    """
    ids = list(records)
    if len(ids) < 2:
        raise ValueError("leave-one-recording-out requires at least two recordings")
    shifts = list(shifts)
    folds: list[dict[str, object]] = []
    for held_out in ids:
        train_ids = [recording_id for recording_id in ids if recording_id != held_out]
        candidates: list[tuple[float, int]] = []
        for shift in shifts:
            values = [
                lag_auc(records[recording_id]["events"], records[recording_id]["score"], shift)
                for recording_id in train_ids
            ]
            values = [value for value in values if value is not None]
            if len(values) == len(train_ids):
                candidates.append((float(np.mean(values)), int(shift)))
        if not candidates:
            folds.append(
                {
                    "held_out": held_out,
                    "train_recordings": train_ids,
                    "selected_shift_frames": 0,
                    "train_mean_auc": None,
                    "held_out_retrospective_auc": None,
                    "held_out_lag0_auc": None,
                    "held_out_prospective_auc": None,
                    "reason": "no shift had a defined AUC on every training recording",
                }
            )
            continue
        train_mean, selected = max(candidates, key=lambda item: (item[0], -abs(item[1]), -item[1]))
        held_events = records[held_out]["events"]
        held_score = records[held_out]["score"]
        retrospective = lag_auc(held_events, held_score, selected)
        lag0 = lag_auc(held_events, held_score, 0)
        prospective_shift = -selected if selected != 0 else -8
        prospective = lag_auc(held_events, held_score, prospective_shift)
        folds.append(
            {
                "held_out": held_out,
                "train_recordings": train_ids,
                "selected_shift_frames": int(selected),
                "selected_shift_ms": None,
                "direction": (
                    "model_lags_event" if selected > 0 else "model_leads_event" if selected < 0 else "synchronous"
                ),
                "train_mean_auc": float(train_mean),
                "held_out_retrospective_auc": retrospective,
                "held_out_lag0_auc": lag0,
                "prospective_shift_frames": int(prospective_shift),
                "held_out_prospective_auc": prospective,
            }
        )
    return folds


def persistence_labels(events: np.ndarray) -> np.ndarray:
    """Predict reversal at ``t`` from the observed event at ``t-1``."""
    events = np.asarray(events, dtype=bool)
    previous = np.zeros_like(events)
    if len(events) > 1:
        previous[1:] = events[:-1]
    return np.where(previous, -1, 1).astype(np.int64)


def majority_labels(events: np.ndarray) -> np.ndarray:
    """Constant majority-class labels; AUC is undefined, agreement is not."""
    prevalence = float(np.mean(events))
    return np.full(len(events), -1 if prevalence > 0.5 else 1, dtype=np.int64)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]


def event_level_metrics(
    events: np.ndarray,
    labels: np.ndarray,
    intervals: Sequence[tuple[int, int]],
    dt: float,
    tolerance_frames: int = 10,
) -> dict[str, object]:
    """Event-level recall, detection latency and false-alarm episodes."""
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    predicted = labels == -1
    detected = 0
    latencies: list[float] = []
    for start, end in intervals:
        lo = max(0, start - tolerance_frames)
        hi = min(n, end + tolerance_frames)
        hits = np.flatnonzero(predicted[lo:hi])
        if len(hits):
            detected += 1
            latencies.append((lo + int(hits[0]) - start) * dt)
    runs = _runs(predicted)
    false_alarms = 0
    for run_start, run_end in runs:
        if not any(run_start < end and run_end > start for start, end in intervals):
            false_alarms += 1
    duration_min = n * dt / 60.0
    return {
        "n_events": len(intervals),
        "n_detected": detected,
        "event_recall": detected / len(intervals) if intervals else None,
        "median_detection_latency_s": float(np.median(latencies)) if latencies else None,
        "n_predicted_runs": len(runs),
        "n_false_alarm_runs": false_alarms,
        "false_alarm_episodes_per_min": false_alarms / duration_min if duration_min > 0 else None,
        "tolerance_frames": int(tolerance_frames),
        "duration_min": float(duration_min),
    }
