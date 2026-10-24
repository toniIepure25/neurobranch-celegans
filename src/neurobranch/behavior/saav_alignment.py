"""Head-curvature phase alignment for the Phase 5 SAAV reconstruction.

The author-local aligned CSVs used by the published SAAV decoder are not
available.  This module builds a transparent raw-data analogue:

* head curvature is the orientation-corrected ``dv`` signal;
* zero crossings define successive half head-swing cycles;
* traces are resampled to four samples per full cycle;
* each event's window ends at its last reversal frame.

The resulting data are a method reconstruction, not an exact reproduction of
the published arrays.  All phase choices are explicit and tested.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from neurobranch.data.kramer import KramerRecording

HC_CHANNEL = "hc"
SAAV_CHANNEL = "saav"
CIRCUIT_LABELS: tuple[str, ...] = (
    "SAAV",
    "SAAD",
    "SMDV",
    "SMDD",
    "RMDV",
    "RMDD",
    "RIV",
    "SMBV",
    "SMBD",
)
DEFAULT_FRAMES = 48
DEFAULT_SAMPLES_PER_CYCLE = 4
MIN_REVERSAL_FRAMES = 12
MIN_ABS_POST_DV = 0.05
MIN_HALF_CYCLE_S = 0.25
MAX_HALF_CYCLE_S = 20.0


@dataclass(frozen=True)
class PhaseTrack:
    """Monotone phase sampled at head-curvature zero crossings."""

    times: np.ndarray
    phases: np.ndarray
    interval_ok: np.ndarray

    def __post_init__(self) -> None:
        if self.times.shape != self.phases.shape or self.times.shape != self.interval_ok.shape:
            raise ValueError("phase track arrays must have matching shapes")
        if len(self.times) < 3:
            raise ValueError("phase track needs at least three crossings")
        if np.any(np.diff(self.times) <= 0) or np.any(np.diff(self.phases) <= 0):
            raise ValueError("phase track must be strictly increasing")


@dataclass
class AlignedDataset:
    """Event-level phase-aligned traces and metadata."""

    x: np.ndarray  # (events, frames, channels)
    y: np.ndarray  # 1 = dorsal, 0 = ventral
    meta: pd.DataFrame
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.x.ndim != 3:
            raise ValueError("x must be (events, frames, channels)")
        if len(self.x) != len(self.y) or len(self.x) != len(self.meta):
            raise ValueError("x, y and meta must have the same event count")
        if self.x.shape[2] != len(self.channels):
            raise ValueError("channel axis does not match channels")

    @property
    def n_events(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.x.shape[1])

    def channel(self, name: str) -> np.ndarray:
        return self.x[:, :, self.channels.index(name)]

    def subset(self, mask: np.ndarray) -> AlignedDataset:
        mask = np.asarray(mask, dtype=bool)
        return AlignedDataset(
            x=self.x[mask],
            y=self.y[mask],
            meta=self.meta.loc[mask].reset_index(drop=True),
            channels=self.channels,
        )


def phase_track(
    curvature: Iterable[float],
    dt: float,
    *,
    smooth_sigma: float = 1.0,
    min_half_cycle_s: float = MIN_HALF_CYCLE_S,
    max_half_cycle_s: float = MAX_HALF_CYCLE_S,
) -> PhaseTrack:
    """Detect curvature zero crossings and assign monotone half-cycle phase.

    A crossing is retained only when its interval from the previous retained
    crossing is physiologically plausible.  Invalid intervals remain in the
    track but are marked in ``interval_ok`` so an event that uses one is
    rejected rather than silently interpolated across it.
    """
    values = np.asarray(curvature, dtype=float).ravel()
    if values.size < 3:
        raise ValueError("curvature needs at least three frames")
    if not np.isfinite(values).all():
        raise ValueError("curvature contains non-finite values")
    if dt <= 0:
        raise ValueError("dt must be positive")

    smooth = (
        gaussian_filter1d(values, sigma=float(smooth_sigma), mode="nearest")
        if smooth_sigma > 0
        else values
    )
    idx = np.flatnonzero(smooth[:-1] * smooth[1:] < 0)
    if idx.size < 3:
        raise ValueError("fewer than three curvature zero crossings")

    times = np.empty(idx.size, dtype=float)
    for k, i in enumerate(idx):
        denom = abs(smooth[i]) + abs(smooth[i + 1])
        frac = 0.5 if denom == 0 else abs(smooth[i]) / denom
        times[k] = (i + frac) * dt
    phases = np.pi * np.arange(idx.size, dtype=float)
    gaps = np.diff(times)
    interval_ok = np.r_[True, (gaps >= min_half_cycle_s) & (gaps <= max_half_cycle_s)]
    return PhaseTrack(times=times, phases=phases, interval_ok=interval_ok)


def phase_at(track: PhaseTrack, t: float) -> float:
    """Return instantaneous phase at time ``t`` (linear within a half-cycle)."""
    t = float(t)
    if t < track.times[0] or t > track.times[-1]:
        raise ValueError("time is outside the phase track")
    j = int(np.searchsorted(track.times, t, side="right") - 1)
    j = min(max(j, 0), len(track.times) - 2)
    frac = (t - track.times[j]) / (track.times[j + 1] - track.times[j])
    return float(track.phases[j] + np.pi * frac)


def phase_times(
    track: PhaseTrack,
    *,
    end_time: float,
    n_frames: int,
    samples_per_cycle: int,
) -> np.ndarray:
    """Return ``n_frames`` times ending at ``end_time`` at uniform phase spacing.

    The last returned time is exactly ``end_time``.  The first is
    ``(n_frames - 1) * 2π / samples_per_cycle`` earlier in phase, not necessarily
    the same elapsed-time distance, because the animal's head-swing frequency
    varies.  Invalid half-cycle intervals are rejected.
    """
    if n_frames < 2:
        raise ValueError("n_frames must be at least two")
    if samples_per_cycle < 2:
        raise ValueError("samples_per_cycle must be at least two")
    step = 2.0 * np.pi / samples_per_cycle
    end_phase = phase_at(track, end_time)
    targets = end_phase - step * np.arange(n_frames - 1, -1, -1, dtype=float)
    if targets[0] < track.phases[0]:
        raise ValueError("not enough phase history before the event")
    out = np.empty(n_frames, dtype=float)
    for k, target in enumerate(targets):
        j = int(np.searchsorted(track.phases, target, side="right") - 1)
        j = min(max(j, 0), len(track.times) - 2)
        if not track.interval_ok[j + 1]:
            raise ValueError("event crosses an implausible head-swing interval")
        frac = (target - track.phases[j]) / (track.phases[j + 1] - track.phases[j])
        out[k] = track.times[j] + frac * (track.times[j + 1] - track.times[j])
    if np.any(np.diff(out) <= 0):
        raise ValueError("phase inversion produced non-monotone times")
    if out[0] < 0 or out[-1] > track.times[-1] + 1e-9:
        raise ValueError("phase window is outside the recording")
    return out


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    sd = float(np.nanstd(values))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(values)
    return (values - mean) / sd


def build_aligned_dataset(
    recordings: list[KramerRecording],
    *,
    cohorts: tuple[str, ...] = ("wt_octanol", "wt_diacetyl"),
    n_frames: int = DEFAULT_FRAMES,
    samples_per_cycle: int = DEFAULT_SAMPLES_PER_CYCLE,
    min_reversal_frames: int = MIN_REVERSAL_FRAMES,
    min_abs_post_dv: float = MIN_ABS_POST_DV,
    require_saav: bool = True,
    neural_channels: tuple[str, ...] = CIRCUIT_LABELS,
) -> tuple[AlignedDataset, pd.DataFrame]:
    """Build phase-aligned event traces from raw Kramer recordings.

    Returns the dataset and an audit table with one row per candidate event and
    an exclusion reason.  Z-scoring is per recording and is explicitly
    retrospective; it must not be used by the online benchmark.
    """
    rows: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    labels: list[int] = []
    channels = (HC_CHANNEL, SAAV_CHANNEL, *[f"neuron:{name}" for name in neural_channels])
    channel_index = {name: i for i, name in enumerate(channels)}

    for rec in recordings:
        if rec.cohort not in cohorts:
            continue
        track: PhaseTrack | None
        try:
            track = phase_track(rec.dv, rec.dt)
        except ValueError:
            track = None
        z_traces = _zscore(rec.traces_raw)
        z_hc = _zscore(rec.dv)
        labelled = rec.identified_indices
        for event_index, (start, end) in enumerate(zip(rec.reversal_onsets, rec.reversal_offsets)):
            start = int(start)
            end = int(end)
            base = {
                "animal_id": rec.animal_id,
                "cohort": rec.cohort,
                "event_index": event_index,
                "start": start,
                "end": end,
                "duration_frames": end - start,
                "included": False,
                "reason": "",
            }
            if end - start < min_reversal_frames:
                rows.append({**base, "reason": "reversal_too_short"})
                continue
            if end + 2 > rec.n_frames:
                rows.append({**base, "reason": "no_post_turn_window"})
                continue
            post_dv = float(np.mean(rec.dv[end : end + 2]))
            if not np.isfinite(post_dv):
                rows.append({**base, "reason": "nonfinite_post_dv"})
                continue
            if abs(post_dv) < min_abs_post_dv:
                rows.append({**base, "reason": "near_zero_turn"})
                continue
            direction = "dorsal" if post_dv > 0 else "ventral"
            y = 1 if direction == "dorsal" else 0
            has_saav = "SAAV" in labelled
            if require_saav and not has_saav:
                rows.append({**base, "reason": "missing_saav_identity"})
                continue
            if track is None:
                rows.append({**base, "reason": "no_phase_track"})
                continue
            try:
                sample_times = phase_times(
                    track,
                    end_time=(end - 1) * rec.dt,
                    n_frames=n_frames,
                    samples_per_cycle=samples_per_cycle,
                )
            except ValueError as exc:
                rows.append({**base, "reason": f"phase:{exc}"})
                continue

            x = np.full((n_frames, len(channels)), np.nan, dtype=float)
            x[:, channel_index[HC_CHANNEL]] = np.interp(sample_times, rec.time_seconds, z_hc)
            if has_saav:
                idx = labelled["SAAV"]
                x[:, channel_index[SAAV_CHANNEL]] = np.interp(
                    sample_times, rec.time_seconds, z_traces[:, idx]
                )
            for name in neural_channels:
                if name in labelled and f"neuron:{name}" in channel_index:
                    idx = labelled[name]
                    x[:, channel_index[f"neuron:{name}"]] = np.interp(
                        sample_times, rec.time_seconds, z_traces[:, idx]
                    )
            if not np.isfinite(x[:, channel_index[HC_CHANNEL]]).all():
                rows.append({**base, "reason": "nonfinite_hc"})
                continue
            if require_saav and not np.isfinite(x[:, channel_index[SAAV_CHANNEL]]).all():
                rows.append({**base, "reason": "nonfinite_saav"})
                continue
            arrays.append(x)
            labels.append(y)
            rows.append(
                {
                    **base,
                    "included": True,
                    "direction": direction,
                    "post_dv": post_dv,
                    "duration_s": (end - start) * rec.dt,
                    "odor_present": bool(np.any(rec.odor_idx[start:end] > 0)),
                    "saav_present": has_saav,
                    "phase_start_s": float(sample_times[0]),
                    "phase_end_s": float(sample_times[-1]),
                }
            )

    audit = pd.DataFrame(rows)
    if not arrays:
        return AlignedDataset(np.empty((0, n_frames, len(channels))), np.empty(0, int), audit, channels), audit
    meta = audit[audit["included"]].reset_index(drop=True).copy()
    return AlignedDataset(np.stack(arrays), np.asarray(labels, dtype=int), meta, channels), audit


def dataset_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    channels: tuple[str, ...],
) -> AlignedDataset:
    """Validate and wrap precomputed arrays (used by tests and cached runs)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int).ravel()
    if x.ndim != 3:
        raise ValueError("x must be three-dimensional")
    if not np.isfinite(x).all():
        raise ValueError("x contains non-finite values")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y must be binary")
    if len(meta) != len(x):
        raise ValueError("meta and x lengths differ")
    return AlignedDataset(x=x, y=y, meta=meta.reset_index(drop=True), channels=tuple(channels))
