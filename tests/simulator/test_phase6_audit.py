from __future__ import annotations

import numpy as np

from neurobranch.dynamics.models import VARModel
from neurobranch.preprocessing import split_starts
from neurobranch.simulator import NeuralSimulator
from neurobranch.simulator.adapters import SupportModel
from neurobranch.simulator.metrics import _autocorr_error


def _var_history_model(toy_data, history: int = 2) -> VARModel:
    starts = split_starts(toy_data, "train", history=history, horizon=1)
    return VARModel.fit(
        toy_data.X,
        toy_data.y_macro,
        starts,
        history=history,
        ridge=1e-2,
        seed=17,
    )


def test_rollout_uses_real_history_not_repeated_x0(toy_data):
    history = 2
    model = _var_history_model(toy_data, history)
    sim = NeuralSimulator(model, toy_data)
    t = int(sim.test_starts(horizon=5, history=history)[0])
    hist = sim.initial_history(t, history=history)
    manual = model.predict_next(hist[None])[0]
    rollout = sim.rollout(None, 1, initial_history=hist)
    assert np.allclose(rollout.states[0], hist[-1])
    assert np.allclose(rollout.states[1], manual)
    assert rollout.metadata["initial_history_padded"] is False

    # Supplying only x0 intentionally pads with x0, and must say so.
    padded = sim.rollout(hist[-1], 1)
    assert padded.metadata["initial_history_padded"] is True
    assert not np.allclose(padded.states[1], manual)


def test_test_starts_include_full_history_window(toy_data):
    history = 2
    model = _var_history_model(toy_data, history)
    sim = NeuralSimulator(model, toy_data)
    starts = sim.test_starts(horizon=5, history=history)
    assert len(starts) > 0
    for t in starts:
        assert toy_data.masks["test"][int(t) - history + 1 : int(t) + 6].all()


def test_rollout_ensemble_records_observed_initial_macro(toy_model, toy_data):
    sim = NeuralSimulator(toy_model, toy_data)
    t = int(sim.test_starts(horizon=3)[0])
    hist = sim.initial_history(t)
    macro = int(toy_data.y_macro[t])
    ens = sim.rollout_ensemble(
        None,
        2,
        3,
        initial_history=hist,
        initial_macro=macro,
        base_seed=1,
    )
    assert ens.initial_macro == macro
    assert np.allclose(ens.x0, hist[-1])


def test_autocorrelation_error_compares_temporal_structure():
    rng = np.random.default_rng(5)
    real = np.cumsum(rng.normal(size=(80, 4)), axis=0)
    assert _autocorr_error(real, real) < 1e-10
    wrong = real.copy()
    wrong[:, 0] = np.random.default_rng(9).permutation(wrong[:, 0])
    assert _autocorr_error(wrong, real) > 0.05


def test_support_threshold_is_calibrated_on_validation(toy_data):
    train = toy_data.X[toy_data.masks["train"]]
    val = toy_data.X[toy_data.masks["val"]]
    support = SupportModel(train, quantile=0.95, calibration_X=val)
    distances = support.score(val)
    assert np.isclose(support.threshold, np.quantile(distances, 0.95))
    assert support.calibration_quantile == 0.95
    assert support.training_threshold > 0
