"""Tests for external recording metadata and leakage-safe adapters."""

from __future__ import annotations

import numpy as np

from neurobranch.data.external import (
    ExternalRecording,
    build_kato_stim_context_worm_data,
    build_kato_stim_worm_data,
    piecewise_oxygen,
)


def _synthetic_recording(n_frames: int = 200, n_traces: int = 10) -> ExternalRecording:
    rng = np.random.default_rng(0)
    labels: list[tuple[str, ...]] = [()] * n_traces
    labels[0] = ("AVAL",)
    labels[1] = ("AVAR", "AVBR")
    labels[2] = ("AFDL", "ASKL")
    time = np.arange(n_frames, dtype=np.float64) / 3.0
    return ExternalRecording(
        source="kato2015_wt_stim",
        session_id="synthetic_0",
        worm_index=0,
        dataset_name="synthetic",
        labels=tuple(labels),
        traces=rng.normal(size=(n_frames, n_traces)),
        raw_traces=rng.normal(size=(n_frames, n_traces)),
        fps=3.0,
        time_seconds=time,
        states=tuple("fwd" for _ in range(n_frames)),
        state_names=("fwd", "rev", "revsus", "turn"),
        stimulus={"identity": "O2", "conc": [4.0, 21.0], "switchtimes": [360.0], "initialstate": [2.0]},
        source_file="synthetic.mat",
        source_sha256="0" * 64,
    )


def test_piecewise_oxygen_toggles_at_each_switch():
    oxygen = piecewise_oxygen(
        np.array([0.0, 359.9, 360.0, 389.9, 390.0, 719.0]),
        np.array([360.0, 390.0]),
        np.array([4.0, 21.0]),
        initialstate=2,
    )
    np.testing.assert_array_equal(oxygen, np.array([21.0, 21.0, 4.0, 4.0, 21.0, 21.0]))


def test_metadata_preserves_multi_neuron_labels_and_behavior_mask():
    rec = _synthetic_recording()
    assert rec.named_mask.sum() == 3
    assert rec.behavior_mask.sum() == 2  # AVAL and the AVBL/AVBR trace
    assert rec.duplicate_label_traces == 2
    assert set(rec.named_labels) == {"AFDL", "ASKL", "AVAL", "AVAR", "AVBR"}
    assert rec.oxygen_percent is not None
    assert set(np.unique(rec.oxygen_percent).tolist()) == {21.0}


def test_stim_adapter_uses_constant_oxygen_segment_and_disjoint_masks():
    rec = _synthetic_recording()
    data = build_kato_stim_worm_data(
        rec,
        max_seconds=60.0,
        train_fraction=0.6,
        val_fraction=0.2,
        purge_frames=5,
    )
    assert data.n_neurons == 8
    assert data.n_frames == 180
    assert len(data.excluded_neurons) == 0  # exclusion happened before WormRecording
    masks = [data.masks[name] for name in ("train", "val", "test")]
    assert not (masks[0] & masks[1]).any()
    assert not (masks[0] & masks[2]).any()
    assert not (masks[1] & masks[2]).any()
    assert np.isfinite(data.X).all()
    # Standardization is defined from the training segment only.
    np.testing.assert_allclose(data.X[data.masks["train"]].mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(data.X[data.masks["train"]].std(axis=0), 1.0, atol=1e-5)
    assert data.y_macro[data.masks["train"]].min() >= 0


def test_stim_adapter_rejects_late_segment_with_changing_oxygen():
    rec = _synthetic_recording()
    try:
        build_kato_stim_worm_data(rec, max_seconds=400.0)
    except ValueError as exc:
        assert "constant" in str(exc)
    else:
        raise AssertionError("expected ValueError for segment crossing the 360 s oxygen switch")


def test_context_adapter_attaches_measured_oxygen_and_keeps_masks_disjoint():
    rec = _synthetic_recording(n_frames=2400)
    data = build_kato_stim_context_worm_data(
        rec,
        train_fraction=0.6,
        val_fraction=0.2,
        purge_frames=5,
    )
    assert data.n_frames == 2400
    assert data.external_input is not None
    assert data.input_name == "oxygen_percent"
    assert np.unique(data.external_input).size == 2
    train = data.masks["train"]
    val = data.masks["val"]
    test = data.masks["test"]
    assert not (train & val).any() and not (train & test).any() and not (val & test).any()
    assert data.external_input.shape == (data.n_frames,)
    # The training block must contain an oxygen change; otherwise B is not
    # identifiable and the experiment should not be called input-conditioned.
    assert np.unique(data.external_input[train]).size == 2


def test_context_adapter_rejects_constant_oxygen():
    rec = _synthetic_recording(n_frames=1200)
    try:
        build_kato_stim_context_worm_data(rec, max_seconds=300.0)
    except ValueError as exc:
        assert "transition" in str(exc)
    else:
        raise AssertionError("expected ValueError without an oxygen transition")


def test_context_adapter_aligns_to_actual_frame_times_not_nominal_30s():
    rec = _synthetic_recording(n_frames=2400)
    data = build_kato_stim_context_worm_data(rec, train_fraction=0.6, val_fraction=0.2)
    change = np.flatnonzero(np.diff(data.external_input) != 0) + 1
    expected = int(np.flatnonzero(rec.time_seconds >= 360.0)[0])
    assert change.size >= 1
    assert int(change[0]) == expected
    assert data.time_seconds is not None
    np.testing.assert_allclose(data.time_seconds, rec.time_seconds)
