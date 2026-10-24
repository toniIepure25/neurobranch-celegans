"""Causally indexed online-prospective turning windows.

The prediction landmark is fixed relative to a causally detected reversal onset.
The eventual reversal end, direction and future neural frames are never used to
build a feature window or choose a landmark.  They are used only to define the
target, the risk set and the observed lead time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from neurobranch.behavior.saav_alignment import CIRCUIT_LABELS, MIN_ABS_POST_DV, MIN_REVERSAL_FRAMES
from neurobranch.data.kato import BEHAVIOR_LABEL_NEURONS
from neurobranch.data.kramer import KramerRecording

OnsetMode = Literal["state", "velocity"]
LANDMARKS_S: tuple[float, ...] = (0.6, 1.2, 2.4, 3.6, 4.8, 7.2)
HISTORY_S = 7.2
MIN_HISTORY_S = 3.6
BEHAVIOR_CHANNELS = ("hc", "velocity", "angular_velocity", "pumping")
SENSORY_CHANNELS = ("odor", "distance")
NEURAL_CHANNELS = ("saav", *[f"neuron:{name}" for name in CIRCUIT_LABELS])


@dataclass
class OnlineDataset:
    x: np.ndarray  # (samples, history_frames, channels), raw (unscaled)
    y: np.ndarray
    meta: pd.DataFrame
    channels: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])

    def channel(self, name: str) -> np.ndarray:
        return self.x[:, :, self.channels.index(name)]

    def subset(self, mask: np.ndarray) -> OnlineDataset:
        mask = np.asarray(mask, dtype=bool)
        return OnlineDataset(
            x=self.x[mask],
            y=self.y[mask],
            meta=self.meta.loc[mask].reset_index(drop=True),
            channels=self.channels,
        )


def detect_velocity_onsets(
    velocity: np.ndarray,
    *,
    threshold: float = -0.01,
    confirmation_frames: int = 2,
    dt: float = 0.6,
) -> list[int]:
    """Detect negative-velocity runs using only current and past frames.

    A detection is returned at the frame where the required confirmation count
    has just become available.  It is therefore causal; no future frame is used.
    """
    values = np.asarray(velocity, dtype=float).ravel()
    if confirmation_frames < 1:
        raise ValueError("confirmation_frames must be positive")
    if not np.isfinite(values).all():
        raise ValueError("velocity contains non-finite values")
    out: list[int] = []
    run = 0
    for i, value in enumerate(values):
        run = run + 1 if value < threshold else 0
        if run == confirmation_frames:
            out.append(i)
    return out


def state_onsets(recording: KramerRecording) -> list[int]:
    """Causal rising edges of the archival reversal state."""
    rv = np.asarray(recording.reversal_vec, dtype=int).ravel()
    if len(rv) != recording.n_frames:
        raise ValueError("reversal_vec length does not match recording")
    previous = np.r_[0, rv[:-1]]
    return [int(i) for i in np.flatnonzero((rv == 1) & (previous == 0))]


def _match_event(
    detected: int,
    onsets: np.ndarray,
    offsets: np.ndarray,
    *,
    tolerance_frames: int = 3,
) -> int | None:
    """Match a detected onset to an annotated event for target lookup only."""
    for idx, (start, end) in enumerate(zip(onsets, offsets)):
        if abs(int(start) - detected) <= tolerance_frames and end > detected:
            return idx
    return None


def _channel_matrix(rec: KramerRecording) -> tuple[np.ndarray, tuple[str, ...]]:
    labelled = rec.identified_indices
    columns = [
        np.asarray(rec.dv, dtype=float),
        np.asarray(rec.velocity, dtype=float),
        np.asarray(rec.angular_velocity, dtype=float),
        np.asarray(rec.pumping, dtype=float),
        np.asarray(rec.odor_idx, dtype=float),
        np.asarray(rec.distance_from_center, dtype=float),
    ]
    names = list(BEHAVIOR_CHANNELS) + list(SENSORY_CHANNELS)
    names.append("saav")
    if "SAAV" in labelled:
        columns.append(np.asarray(rec.traces_raw[:, labelled["SAAV"]], dtype=float))
    else:
        columns.append(np.full(rec.n_frames, np.nan))
    for name in CIRCUIT_LABELS:
        names.append(f"neuron:{name}")
        if name in labelled:
            columns.append(np.asarray(rec.traces_raw[:, labelled[name]], dtype=float))
        else:
            columns.append(np.full(rec.n_frames, np.nan))
    if labelled:
        all_population = np.column_stack([rec.traces_raw[:, i] for i in labelled.values()])
        all_mean = np.nanmean(all_population, axis=1)
        all_std = np.nanstd(all_population, axis=1)
        clean = [
            rec.traces_raw[:, i]
            for name, i in labelled.items()
            if name not in BEHAVIOR_LABEL_NEURONS
            and not name.startswith(("SMD", "AVA", "RIB"))
            and name != "SAAV"
        ]
        if clean:
            clean_population = np.column_stack(clean)
            columns.extend(
                [
                    np.nanmean(clean_population, axis=1),
                    np.nanstd(clean_population, axis=1),
                    all_mean,
                    all_std,
                ]
            )
        else:
            columns.extend([np.full(rec.n_frames, np.nan)] * 4)
    else:
        columns.extend([np.full(rec.n_frames, np.nan)] * 4)
    names.extend(["population_mean", "population_std", "population_all_mean", "population_all_std"])
    return np.column_stack(columns), tuple(names)


def build_online_dataset(
    recordings: list[KramerRecording],
    *,
    onset_mode: OnsetMode = "state",
    landmarks_s: tuple[float, ...] = LANDMARKS_S,
    history_s: float = HISTORY_S,
    min_history_s: float = MIN_HISTORY_S,
    cohorts: tuple[str, ...] = ("wt_octanol", "wt_diacetyl"),
    min_reversal_frames: int = MIN_REVERSAL_FRAMES,
) -> tuple[OnlineDataset, pd.DataFrame]:
    """Build one causal window per event × landmark.

    The returned audit records every candidate event/landmark that was excluded
    and why.  The primary dataset includes all wild-type events, including those
    without SAAV; neural modes must subset on ``saav_available``.
    """
    rows: list[dict[str, Any]] = []
    x_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    channels: tuple[str, ...] | None = None
    history_frames = int(round(history_s)) if history_s < 2 else int(round(history_s / 0.6))
    # The recordings use a nominal 0.6 s frame.  Use each recording's own dt for
    # landmark conversion but a common nominal history length for batching.
    history_frames = max(4, history_frames)

    for rec in recordings:
        if rec.cohort not in cohorts:
            continue
        matrix, names = _channel_matrix(rec)
        if channels is None:
            channels = names
        elif channels != names:
            raise ValueError("channel schema changed across recordings")
        if onset_mode == "state":
            detected = state_onsets(rec)
        elif onset_mode == "velocity":
            detected = detect_velocity_onsets(rec.velocity, dt=rec.dt)
        else:
            raise ValueError(f"unknown onset_mode {onset_mode!r}")

        for t0 in detected:
            event_index = _match_event(int(t0), rec.reversal_onsets, rec.reversal_offsets)
            if event_index is None:
                rows.append(
                    {
                        "animal_id": rec.animal_id,
                        "cohort": rec.cohort,
                        "detected_onset": int(t0),
                        "event_index": -1,
                        "included": False,
                        "reason": "unmatched_onset",
                    }
                )
                continue
            start = int(rec.reversal_onsets[event_index])
            end = int(rec.reversal_offsets[event_index])
            if end - start < min_reversal_frames:
                rows.append(
                    {
                        "animal_id": rec.animal_id,
                        "cohort": rec.cohort,
                        "detected_onset": int(t0),
                        "event_index": event_index,
                        "included": False,
                        "reason": "reversal_too_short",
                    }
                )
                continue
            if end + 2 > rec.n_frames:
                rows.append(
                    {
                        "animal_id": rec.animal_id,
                        "cohort": rec.cohort,
                        "detected_onset": int(t0),
                        "event_index": event_index,
                        "included": False,
                        "reason": "no_post_window",
                    }
                )
                continue
            post_dv = float(np.mean(rec.dv[end : end + 2]))
            if not np.isfinite(post_dv):
                rows.append(
                    {
                        "animal_id": rec.animal_id,
                        "cohort": rec.cohort,
                        "detected_onset": int(t0),
                        "event_index": event_index,
                        "included": False,
                        "reason": "nonfinite_post_dv",
                    }
                )
                continue
            if abs(post_dv) < MIN_ABS_POST_DV:
                rows.append(
                    {
                        "animal_id": rec.animal_id,
                        "cohort": rec.cohort,
                        "detected_onset": int(t0),
                        "event_index": event_index,
                        "included": False,
                        "reason": "near_zero_turn",
                    }
                )
                continue
            direction = "dorsal" if post_dv > 0 else "ventral"
            y = 1 if direction == "dorsal" else 0
            saav_available = "SAAV" in rec.identified_indices
            for landmark in landmarks_s:
                t_pred = int(round(t0 + landmark / rec.dt))
                base = {
                    "animal_id": rec.animal_id,
                    "event_uid": f"{rec.animal_id}:{event_index}",
                    "cohort": rec.cohort,
                    "event_index": event_index,
                    "detected_onset": int(t0),
                    "start": start,
                    "end": end,
                    "landmark_s": float(landmark),
                    "prediction_frame": int(t_pred),
                    "post_dv": post_dv,
                    "direction": direction,
                    "saav_available": saav_available,
                    "included": False,
                    "reason": "",
                }
                if t_pred >= end:
                    rows.append({**base, "reason": "reversal_ended_before_landmark"})
                    continue
                first = t_pred - history_frames + 1
                if first < 0:
                    rows.append({**base, "reason": "insufficient_recording_history"})
                    continue
                if (t_pred - first + 1) * rec.dt < min_history_s:
                    rows.append({**base, "reason": "insufficient_min_history"})
                    continue
                window = matrix[first : t_pred + 1].T  # channels × frames
                # Store samples as frames × channels for the sequence model.
                x = window.T
                if not np.isfinite(x[:, : len(BEHAVIOR_CHANNELS) + len(SENSORY_CHANNELS)]).all():
                    rows.append({**base, "reason": "nonfinite_behavior"})
                    continue
                x_parts.append(x.astype(np.float32))
                y_parts.append(y)
                rows.append(
                    {
                        **base,
                        "included": True,
                        "lead_time_s": float((end - t_pred) * rec.dt),
                        "risk_set_duration_s": float((end - t0) * rec.dt),
                    }
                )

    audit = pd.DataFrame(rows)
    if not x_parts or channels is None:
        return OnlineDataset(np.empty((0, history_frames, len(channels or ())), np.float32), np.empty(0, int), audit, channels or ()), audit
    meta = pd.DataFrame([r for r in rows if r.get("included")]).reset_index(drop=True)
    return OnlineDataset(np.stack(x_parts), np.asarray(y_parts, dtype=int), meta, channels), audit


def risk_set_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Counts of usable events by landmark, direction and exclusion reason."""
    included = audit[audit["included"]].copy()
    if included.empty:
        return pd.DataFrame(columns=["landmark_s", "direction", "n_events", "n_animals"])
    out = (
        included.groupby(["landmark_s", "direction"])
        .agg(n_events=("event_index", "size"), n_animals=("animal_id", "nunique"))
        .reset_index()
    )
    return out


def excluded_reason_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Exclusion counts by landmark; useful for risk-set selection disclosure."""
    if audit.empty:
        return pd.DataFrame(columns=["landmark_s", "reason", "n"])
    frame = audit.copy()
    frame["landmark_s"] = frame["landmark_s"].fillna(-1.0)
    return frame.groupby(["landmark_s", "reason"]).size().reset_index(name="n")
