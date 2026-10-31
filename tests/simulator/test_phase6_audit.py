from __future__ import annotations

import numpy as np

from neurobranch.dynamics.checkpointing import load_model, save_model
from neurobranch.dynamics.models import VARModel
from neurobranch.dynamics.switching import SwitchingVARModel
from neurobranch.preprocessing import split_starts
from neurobranch.simulator import NeuralSimulator
from neurobranch.simulator.adapters import SupportModel
from neurobranch.simulator.branching import markov_local, markov_transition_matrix
from neurobranch.simulator.metrics import _autocorr_error, distribution_metrics


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


def test_stochastic_var_has_native_innovations_and_spectral_cap(toy_data):
    starts = split_starts(toy_data, "train", history=1, horizon=1)
    model = VARModel.fit(
        toy_data.X,
        toy_data.y_macro,
        starts,
        history=1,
        ridge=1e-2,
        spectral_cap=0.95,
        noise_model="gaussian",
        seed=17,
    )
    assert model.native_stochastic
    assert model.noise_cov is not None
    assert model.spectral_radius <= 0.95 + 1e-8
    hist = toy_data.X[int(starts[0])][None]
    a = model.rollout(hist[-1], 3, seed=1, initial_history=hist, stochastic=True)
    b = model.rollout(hist[-1], 3, seed=2, initial_history=hist, stochastic=True)
    assert not np.allclose(a, b)
    assert np.isfinite(a).all()


def test_switching_var_fit_is_finite_and_has_occupancy(toy_data):
    starts = split_starts(toy_data, "train", history=1, horizon=1)
    model = SwitchingVARModel.fit(
        toy_data.X,
        toy_data.y_macro,
        starts,
        K=2,
        ridge=1e-2,
        seed=17,
        restarts=1,
        n_iter=5,
    )
    assert model.A.shape == (2, toy_data.n_neurons, toy_data.n_neurons)
    assert np.isclose(model.stationary.sum(), 1.0)
    assert (model.stationary >= 0).all()
    hist = toy_data.X[int(starts[0])][None]
    a = model.rollout(hist[-1], 3, seed=1, initial_history=hist, stochastic=True)
    b = model.rollout(hist[-1], 3, seed=2, initial_history=hist, stochastic=True)
    assert np.isfinite(a).all()
    assert not np.allclose(a, b)


def test_covariance_error_is_zero_for_identical_sequences():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 5))
    assert distribution_metrics(x, x)["covariance_error"] < 1e-12


def test_switching_checkpoint_roundtrip(toy_data, tmp_path):
    starts = split_starts(toy_data, "train", history=1, horizon=1)
    model = SwitchingVARModel.fit(
        toy_data.X, toy_data.y_macro, starts, K=2, seed=5, restarts=1, n_iter=3
    )
    path = tmp_path / "switching.pt"
    save_model(model, path, metadata={"phase": 6})
    restored, metadata = load_model(path)
    assert metadata["phase"] == 6
    assert restored.diagonal is True
    assert np.allclose(restored.A, model.A)
    assert np.allclose(restored.P, model.P)


def test_markov_baseline_is_row_stochastic_and_normalized(toy_data):
    transition = markov_transition_matrix(toy_data)
    assert transition.shape == (4, 4)
    assert np.allclose(transition.sum(axis=1), 1.0)
    p = markov_local(transition, 0, horizon=5, n_chains=200, seed=1)
    assert p.shape == (4,)
    assert np.isclose(p.sum(), 1.0)
    assert (p >= 0).all()
