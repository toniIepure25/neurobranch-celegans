"""Tests for causal online landmarks and future-information exclusion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.behavior.online_turning import (
    _channel_matrix,
    build_online_dataset,
    detect_velocity_onsets,
    state_onsets,
)
from neurobranch.data.kramer import KramerRecording


def _recording() -> KramerRecording:
    n = 100
    t = np.arange(n) * 0.6
    velocity = np.zeros(n)
    velocity[20:30] = -0.08
    rv = np.zeros(n, dtype=int)
    rv[20:30] = 1
    dv = np.sin(np.arange(n) / 4)
    traces = np.column_stack([np.cos(np.arange(n) / 5), np.sin(np.arange(n) / 7)])
    return KramerRecording(
        animal_id="synthetic",
        cohort="wt_octanol",
        source_mat="",
        source_h5="",
        source_sha256="",
        traces_raw=traces,
        traces_published_z=traces,
        time_seconds=t,
        velocity=velocity,
        reversal_vec=rv,
        reversal_onsets=np.array([20]),
        reversal_offsets=np.array([30]),
        head_angle=dv,
        dv=dv,
        worm_angle=np.zeros(n),
        angular_velocity=np.zeros(n) + 0.01,
        pumping=np.zeros(n) + 1.0,
        distance_from_center=np.zeros(n) + 2.0,
        odor_idx=np.zeros(n, dtype=int),
        labels=("SAAV", None),
        label_ambiguous=(False, False),
    )


def test_detect_velocity_onsets_is_confirmation_causal() -> None:
    velocity = np.array([0.1, -0.02, -0.03, 0.1, -0.04])
    onsets = detect_velocity_onsets(velocity, threshold=-0.01, confirmation_frames=2)
    assert onsets == [2]


def test_state_onsets_uses_rising_edges() -> None:
    rec = _recording()
    assert state_onsets(rec) == [20]


def test_online_window_excludes_future_frames() -> None:
    rec = _recording()
    data, audit = build_online_dataset(
        [rec],
        landmarks_s=(0.6,),
        history_s=3.6,
        min_history_s=3.6,
        min_reversal_frames=5,
    )
    assert data.n_samples == 1
    assert data.meta.iloc[0]["prediction_frame"] == 21
    original = data.x.copy()
    # Change everything after the prediction; the feature window must not move.
    rec_future = _recording()
    rec_future.dv[22:] += 100.0
    rec_future.traces_raw[22:, 0] += 100.0
    data_future, _ = build_online_dataset(
        [rec_future],
        landmarks_s=(0.6,),
        history_s=3.6,
        min_history_s=3.6,
        min_reversal_frames=5,
    )
    assert np.array_equal(original, data_future.x, equal_nan=True)
    assert audit.iloc[0]["included"] is True or bool(audit.iloc[0]["included"])


def test_online_risk_set_excludes_short_reversal_at_late_landmark() -> None:
    rec = _recording()
    data, audit = build_online_dataset(
        [rec],
        landmarks_s=(0.6, 7.2),
        history_s=3.6,
        min_history_s=3.6,
        min_reversal_frames=5,
    )
    included = set(data.meta["landmark_s"].tolist())
    assert 0.6 in included
    assert 7.2 not in included
    assert "reversal_ended_before_landmark" in set(audit["reason"])


def test_online_audit_has_no_nan_in_causal_feature_columns() -> None:
    rec = _recording()
    data, _ = build_online_dataset(
        [rec],
        landmarks_s=(0.6,),
        history_s=3.6,
        min_history_s=3.6,
        min_reversal_frames=5,
    )
    frame = pd.DataFrame(data.x[:, :, :4].reshape(data.n_samples, -1))
    assert np.isfinite(frame.to_numpy()).all()


def test_direction_label_orientation_uses_post_dv() -> None:
    dorsal = _recording()
    dorsal.dv[30:32] = 0.5
    ventral = _recording()
    ventral.dv[30:32] = -0.5
    d_data, _ = build_online_dataset(
        [dorsal], landmarks_s=(0.6,), history_s=3.6, min_history_s=3.6, min_reversal_frames=5
    )
    v_data, _ = build_online_dataset(
        [ventral], landmarks_s=(0.6,), history_s=3.6, min_history_s=3.6, min_reversal_frames=5
    )
    assert d_data.y[0] == 1
    assert v_data.y[0] == 0
    assert d_data.meta.iloc[0]["direction"] == "dorsal"
    assert v_data.meta.iloc[0]["direction"] == "ventral"


def test_missing_identity_is_nan_not_imputed_in_raw_online_data() -> None:
    data, _ = build_online_dataset([_recording()], landmarks_s=(0.6,), history_s=3.6, min_history_s=3.6, min_reversal_frames=5)
    assert np.isnan(data.channel("neuron:SAAD")).all()
    assert np.isfinite(data.channel("saav")).all()


def test_population_excludes_behaviour_label_and_saav_neurons() -> None:
    rec = _recording()
    n = rec.n_frames
    rec.traces_raw = np.column_stack([np.full(n, 1.0), np.full(n, 2.0), np.full(n, 3.0)])
    rec.labels = ("SAAV", "SMDD", "AWCL")
    rec.label_ambiguous = (False, False, False)
    matrix, names = _channel_matrix(rec)
    clean_mean = matrix[:, names.index("population_mean")]
    all_mean = matrix[:, names.index("population_all_mean")]
    assert np.allclose(clean_mean, 3.0)
    assert np.allclose(all_mean, 2.0)
