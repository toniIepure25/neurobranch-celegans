"""Tests for head-curvature phase alignment and reconstructed-data guards."""

from __future__ import annotations

import numpy as np
import pytest

from neurobranch.behavior.saav_alignment import (
    PhaseTrack,
    build_aligned_dataset,
    dataset_from_arrays,
    phase_at,
    phase_times,
    phase_track,
)


def _sine_track() -> PhaseTrack:
    dt = 0.1
    t = np.arange(0, 100, dt)
    curvature = np.sin(2 * np.pi * 0.2 * t)
    return phase_track(curvature, dt, smooth_sigma=0.0)


def test_phase_track_is_monotone_and_starts_at_crossing() -> None:
    track = _sine_track()
    assert len(track.times) > 20
    assert np.all(np.diff(track.times) > 0)
    assert np.allclose(np.diff(track.phases), np.pi)
    assert track.interval_ok[0]


def test_phase_times_end_at_prediction_time_and_uniform_phase() -> None:
    track = _sine_track()
    end = 40.05
    times = phase_times(track, end_time=end, n_frames=12, samples_per_cycle=4)
    assert times[-1] == pytest.approx(end)
    phases = [phase_at(track, t) for t in times]
    steps = np.diff(phases)
    assert np.allclose(steps, np.pi / 2, atol=1e-6)
    assert np.all(np.diff(times) > 0)


def test_phase_times_reject_insufficient_history() -> None:
    track = _sine_track()
    with pytest.raises(ValueError):
        phase_times(track, end_time=0.3, n_frames=48, samples_per_cycle=4)


def test_phase_track_marks_implausible_half_cycles() -> None:
    dt = 0.1
    t = np.arange(0, 100, dt)
    curvature = np.sin(2 * np.pi * 0.2 * t)
    curvature[500:800] = 0.0  # flat region removes crossings and creates a long gap
    track = phase_track(curvature, dt, smooth_sigma=0.0)
    assert not np.all(track.interval_ok)


def test_dataset_from_arrays_rejects_nonfinite_and_mismatch() -> None:
    import pandas as pd

    x = np.zeros((3, 4, 1))
    y = np.array([0, 1, 0])
    meta = pd.DataFrame({"animal_id": ["a", "a", "b"]})
    dataset_from_arrays(x, y, meta, ("hc",))
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        dataset_from_arrays(x, y, meta, ("hc",))
    with pytest.raises(ValueError):
        dataset_from_arrays(np.zeros((3, 4, 1)), y, meta.iloc[:2], ("hc",))


def test_build_aligned_dataset_without_arrays_returns_empty() -> None:
    dataset, audit = build_aligned_dataset([])
    assert dataset.n_events == 0
    assert audit.empty
