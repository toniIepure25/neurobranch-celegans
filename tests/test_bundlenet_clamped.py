"""Contract tests for the published 15-intrinsic/69-signal clamped core."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurobranch.bundlenet_sim.clamped import (
    ClampedParameters,
    SignalPulse,
    simulate_clamped,
)
from neurobranch.bundlenet_sim.premotor import load_premotor
from neurobranch.bundlenet_sim.recording import (
    ClampedDataError,
    build_clamped_recording,
    load_atanas_recording,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT
    / "references"
    / "external"
    / "Celegans_premotor"
    / "data"
    / "Atanas_data"
)

pytestmark = pytest.mark.skipif(
    not DATA.exists() or not (ROOT / "data" / "processed" / "premotor" / "W.npy").exists(),
    reason="run scripts/fetch_vienna_artifacts.py to restore premotor references",
)


def _recording(recording_id: str = "2023-01-23-15"):
    model = load_premotor()
    atanas = load_atanas_recording(DATA / f"{recording_id}.json")
    return model, build_clamped_recording(model, atanas, expected_core=None)


def test_canonical_recording_has_published_counts() -> None:
    _, recording = _recording()
    assert recording.n_core == 84
    assert recording.n_intrinsic == 15
    assert recording.n_signal == 69
    assert recording.replacement_map == {"AVBR": "AVBL", "RIBR": "AVBL"}
    assert set(np.unique(recording.behavior)).issubset({-1, 0, 1})


def test_every_catalog_recording_has_all_intrinsic_neurons() -> None:
    model = load_premotor()
    valid = []
    for path in sorted(DATA.glob("*.json")):
        try:
            recording = build_clamped_recording(
                model, load_atanas_recording(path), expected_core=None
            )
        except ClampedDataError:
            continue
        valid.append(recording)
        assert recording.n_intrinsic == 15
        assert recording.n_core >= 15
        assert recording.n_frames == 1401
    assert len(valid) >= 1


def test_expected_core_rejects_partial_recording() -> None:
    model = load_premotor()
    atanas = load_atanas_recording(DATA / "2023-01-09-15.json")
    with pytest.raises(ClampedDataError, match="expected 84 core neurons"):
        build_clamped_recording(model, atanas, expected_core=84)


def test_clamped_simulation_is_finite_and_well_shaped() -> None:
    model, recording = _recording()
    result = simulate_clamped(model, recording, ClampedParameters())
    assert result.intrinsic.shape == (15, 1401)
    assert result.signals.shape == (69, 1401)
    assert result.full.shape == (84, 1401)
    assert result.status in {"ok", "unstable"}
    assert np.isfinite(result.intrinsic).all()
    assert result.metrics["finite"] is True
    assert 0.0 <= result.metrics["state_agreement"] <= 1.0


def test_signal_pulse_changes_trajectory() -> None:
    model, recording = _recording()
    base = simulate_clamped(model, recording, ClampedParameters())
    pulsed = simulate_clamped(
        model,
        recording,
        ClampedParameters(
            signal_pulses=(
                SignalPulse("RIS", amplitude=10.0, start_s=0.5, duration_s=4.0),
            )
        ),
    )
    assert np.isfinite(pulsed.intrinsic).all()
    assert not np.allclose(base.signals, pulsed.signals)
    assert not np.allclose(base.intrinsic, pulsed.intrinsic)


def test_beta_zero_is_stable_control() -> None:
    model, recording = _recording()
    result = simulate_clamped(model, recording, ClampedParameters(beta=0.0))
    assert np.isfinite(result.intrinsic).all()
    assert float(np.max(np.abs(result.intrinsic))) <= result.parameters.clip
