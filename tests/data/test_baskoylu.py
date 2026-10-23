from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from neurobranch.behavior.prospective import (
    _causal_shift,
    _metrics,
    behavior_matrix,
    causal_zscore,
    chronological_split,
    eligible_frames,
    horizon_frames,
    horizon_labels,
)
from neurobranch.data.baskoylu import load_recording


def _write_h5(path: Path, *, rv: np.ndarray, events: np.ndarray, n_neurons: int = 4) -> None:
    t = 50
    rng = np.random.default_rng(2)
    with h5py.File(path, "w") as f:
        g = f.create_group("gcamp")
        g.create_dataset("trace_array_original", data=rng.normal(size=(t, n_neurons)))
        g.create_dataset("trace_array", data=rng.normal(size=(t, n_neurons)))
        timing = f.create_group("timing")
        timing.create_dataset("timestamp_confocal", data=np.arange(t) * 0.6)
        b = f.create_group("behavior")
        b.create_dataset("velocity", data=np.linspace(-1, 1, t))
        b.create_dataset("reversal_vec", data=rv.astype(np.int64))
        b.create_dataset("reversal_events", data=events.T.astype(np.int64))
        b.create_dataset("head_angle", data=np.zeros(t))
        b.create_dataset("angular_velocity", data=np.zeros(t))
        b.create_dataset("dist_from_cuCenter", data=np.linspace(100, 200, t))


@pytest.fixture()
def recording(tmp_path: Path):
    rv = np.zeros(50, dtype=int)
    rv[0:3] = 1  # partial initial reversal, omitted from the event table
    rv[10:14] = 1
    events = np.array([[1, 3], [10, 14]])
    p = tmp_path / "2024-01-01-01-cudata.h5"
    _write_h5(p, rv=rv, events=events)
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "animal_id\ttrace_index\tlabel\tneuron_class\tconfidence\n"
        "2024-01-01-01\t0\tAVAL\tAVA\t5\n"
        "2024-01-01-01\t1\tAVAR\tAVA\t3\n"
        "2024-01-01-01\t2\tAVAR\tAVA\t2\n"
        "2024-01-01-01\t3\tRIML\tRIM\t4\n",
        encoding="utf-8",
    )
    return load_recording(p, labels_path=labels)


def test_event_indices_are_zero_based_exclusive(recording):
    assert recording.reversal_onsets.tolist() == [1, 10]
    assert recording.reversal_offsets.tolist() == [3, 14]
    assert recording.quality["initial_partial_reversal"] is True
    assert recording.quality["event_vec_mismatch_frames"] == 1


def test_timestamp_alignment(recording):
    assert recording.n_frames == 50
    assert recording.dt == pytest.approx(0.6)
    assert recording.time_seconds[0] == 0.0
    assert recording.time_seconds[-1] == pytest.approx(29.4)
    assert horizon_frames(recording, 6.0) == 10


def test_identity_mapping_and_ambiguity(recording):
    assert recording.labels[:4] == ("AVAL", "AVAR", "AVAR", "RIML")
    assert recording.label_ambiguous[:4] == (False, True, True, False)
    # Duplicate identities are excluded, never silently collapsed.
    assert set(recording.identified_indices) == {"AVAL", "RIML"}


def test_bad_event_is_rejected(tmp_path: Path):
    p = tmp_path / "2024-01-01-02-cudata.h5"
    rv = np.zeros(50, dtype=int)
    rv[10:14] = 1
    _write_h5(p, rv=rv, events=np.array([[9, 14]]))
    with pytest.raises(ValueError, match="non-reversal"):
        load_recording(p, labels_path=tmp_path / "missing.tsv")


def test_event_vector_disagreement_is_rejected(tmp_path: Path):
    p = tmp_path / "2024-01-01-03-cudata.h5"
    rv = np.zeros(50, dtype=int)
    rv[10:14] = 1
    _write_h5(p, rv=rv, events=np.array([[11, 14]]))
    with pytest.raises(ValueError, match="disagree"):
        load_recording(p, labels_path=tmp_path / "missing.tsv")


def test_horizon_labels_use_only_future_onsets(recording):
    # Replace the event table with a single onset at frame 20 for this unit test.
    onset = np.zeros(50, dtype=np.int8)
    onset[20] = 1
    recording.reversal_onsets = np.array([20])
    h = 2
    idx = np.array([17, 18, 19])
    y = horizon_labels(recording, idx, h)
    assert y.tolist() == [0, 1, 1]  # 20 is within (18,20] and (19,21], not (17,19]
    # A frame whose event is at t+h+1 is outside the horizon.
    assert horizon_labels(recording, np.array([17]), 2)[0] == 0


def test_current_reversal_is_ineligible(recording):
    h = 5
    idx = eligible_frames(recording, h)
    assert not np.any(recording.reversal_vec[idx] == 1)
    assert np.all(idx + h < recording.n_frames)


def test_causal_zscore_does_not_use_future():
    x = np.arange(30, dtype=float)[:, None]
    z = causal_zscore(x)
    changed = x.copy()
    changed[15:] = 1e6
    z2 = causal_zscore(changed)
    assert np.allclose(z[:15], z2[:15])


def test_causal_shift_does_not_wrap_future_to_start():
    x = np.arange(20, dtype=float)[:, None]
    shifted = _causal_shift(x, frames=5)
    assert np.allclose(shifted[:5], 0.0)
    assert np.allclose(shifted[5:, 0], x[:-5, 0])


def test_behavior_features_do_not_use_future(recording):
    b, _ = behavior_matrix(recording)
    changed = recording
    changed.velocity = recording.velocity.copy()
    changed.velocity[30:] = 1e6
    b2, _ = behavior_matrix(changed)
    assert np.allclose(b[:30], b2[:30])


def test_chronological_split_has_purge_gap():
    train, val, test = chronological_split(200, h=10)
    assert train[-1] < val[0] - 9
    assert val[-1] < test[0] - 9
    assert len(np.intersect1d(train, test)) == 0


def test_calibration_diagnostics_are_finite():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    p = np.array([0.05, 0.15, 0.2, 0.8, 0.9, 0.7, 0.6, 0.3, 0.55, 0.4])
    m = _metrics(y, p)
    assert np.isfinite(m["calibration_slope"])
    assert np.isfinite(m["calibration_intercept"])
    assert m["log_loss"] > 0
    assert 0 <= m["brier"] <= 1
