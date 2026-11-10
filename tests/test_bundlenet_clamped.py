"""Contract tests for the published 15-intrinsic/69-signal clamped core."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurobranch.bundlenet_sim.clamped import (
    ClampedParameters,
    SignalPulse,
    _rhs_factory,
    simulate_clamped,
)
from neurobranch.bundlenet_sim.premotor import load_premotor
from neurobranch.bundlenet_sim.recording import (
    ClampedDataError,
    build_clamped_recording,
    load_atanas_recording,
    paper_replacement,
    smooth_traces,
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
    assert recording.replacement_map == {"AVBR": "AVBL", "RIBR": "RIBL"}
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


def test_A_orientation_matches_julia_reference() -> None:
    """The h5py/JLD2 layout must be transposed back to Julia's orientation."""
    model = load_premotor()
    names = model.neuron_names
    aval, avar = names.index("AVAL"), names.index("AVAR")
    assert model.A[avar, aval] == pytest.approx(-1.2104317919073022, rel=0, abs=1e-12)
    assert model.A[aval, avar] == pytest.approx(34.0679009535433, rel=0, abs=1e-12)


def test_rhs_at_t0_matches_julia_reference() -> None:
    """Guard both the A orientation and the gap-junction sign against regressions."""
    model, recording = _recording()
    core = np.array([model.neuron_names.index(n) for n in recording.core_names])
    A = model.A[np.ix_(core, core)]
    d = model.d[core]
    W = model.W[np.ix_(core, core)]
    rhs, ids_intrinsic, _, _ = _rhs_factory(
        A, d, W, recording, ClampedParameters()
    )
    du0 = rhs(0.0, recording.traces[ids_intrinsic, 0])
    expected = np.array(
        [
            -16.083109782464533,
            112.64385870499271,
            77.9646362853901,
            97.23485913462397,
            -0.6147477633606113,
            335.3359955947053,
            -99.72632636151721,
            -141.99687243911163,
            218.13014759131036,
            -351.5885497486995,
            -114.35425375437255,
            -192.52128481036345,
            228.05528580984605,
            -41.61818374351512,
            6.684886121825482,
        ]
    )
    np.testing.assert_allclose(du0, expected, rtol=1e-9, atol=1e-9)


def test_external_behavior_validation_report_is_consistent() -> None:
    from neurobranch.bundlenet_sim.workbench import ViennaWorkbench

    report = ViennaWorkbench().tier3_behavior_validation()
    if not report:
        pytest.skip("run scripts/build_tier3_behavior_validation.py first")
    aggregate = report["aggregate"]
    assert aggregate["n_recordings"] == 3
    assert aggregate["mean_readout_auc"] > aggregate["mean_model_auc"]
    assert aggregate["mean_readout_agreement"] > aggregate["mean_model_agreement"]


def test_behavior_readout_uses_published_forward_and_reverse_groups() -> None:
    from neurobranch.bundlenet_sim.recording import INTRINSIC_NEURONS, behavior_readout

    traces = np.zeros((len(INTRINSIC_NEURONS), 3), dtype=float)
    idx = {name: i for i, name in enumerate(INTRINSIC_NEURONS)}
    traces[idx["AVBL"]] = 6.0
    traces[idx["AVAL"]] = 2.0
    labels = behavior_readout(traces, INTRINSIC_NEURONS)
    # AVBL drives the forward group; AVAL partially cancels the reverse group.
    assert labels[0] == 1
    traces[idx["AVAL"]] = 12.0
    labels = behavior_readout(traces, INTRINSIC_NEURONS)
    assert labels[0] == -1


def test_aggregate_julia_parity_report_covers_three_recordings() -> None:
    from neurobranch.bundlenet_sim.workbench import ViennaWorkbench

    parity = ViennaWorkbench().tier3_reference_all()
    if not parity:
        pytest.skip("run scripts/compare_tier3_reference.py --all first")
    aggregate = parity["aggregate"]
    assert aggregate["n_recordings"] == 3
    assert aggregate["mean_pearson_r"] > 0.999
    assert aggregate["mean_behavior_port_agreement"] > 0.999


def test_atanas_labels_use_julia_one_based_indices(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.json"
    path.write_text(
        json.dumps(
            {
                "trace_array": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                "avg_timestep": 0.01,
                "labeled": {
                    "1": {"label": "AVAL"},
                    "2": {"label": "AVAR"},
                },
            }
        ),
        encoding="utf-8",
    )
    atanas = load_atanas_recording(path)
    assert atanas.labels == ("AVAL", "AVAR", None)
    assert np.allclose(atanas.traces[0], [1.0, 2.0])


def test_smoothing_matches_julia_edge_crop() -> None:
    traces = np.arange(40, dtype=float).reshape(2, 20)
    smoothed = smooth_traces(traces)
    assert smoothed.shape == (2, 18)


def test_replacement_order_matches_julia() -> None:
    available = {"AVBL", "RIBL", "RIML", "RIMR"}
    assert paper_replacement("AVBR", available) == "AVBL"
    assert paper_replacement("RIBR", available) == "RIBL"
    assert paper_replacement("AIBL", available) == "RIML"
    assert paper_replacement("RID", available) == "AVBL"


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
