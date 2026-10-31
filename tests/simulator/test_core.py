from __future__ import annotations

import numpy as np
import pytest

from neurobranch.simulator import (
    NeuralSimulator,
    PerturbationSpec,
)
from neurobranch.simulator.metrics import one_step_metrics, rollout_metrics


def test_rollout_is_free_running_and_starts_at_x0(toy_model, toy_data):
    sim = NeuralSimulator(toy_model, toy_data)
    x0 = toy_data.X[int(sim.test_starts(horizon=5)[0])].copy()
    out = sim.rollout(x0, 5, seed=3, noise_std=0.0)
    assert out.states.shape == (6, toy_data.n_neurons)
    assert np.allclose(out.states[0], x0)
    assert out.behavior_proba.shape == (5, 4)
    assert out.finite
    assert out.metadata["support_threshold"] > 0


def test_rollout_does_not_read_future_observations(toy_model, toy_data):
    sim = NeuralSimulator(toy_model, toy_data)
    t = int(sim.test_starts(horizon=5)[0])
    x0 = toy_data.X[t].copy()
    first = sim.rollout(x0, 5, seed=11).states.copy()
    # Corrupt every future real frame; only x0 and the model may be used.
    original = toy_data.X.copy()
    try:
        toy_data.X[t + 1 : t + 30] = np.nan
        second = sim.rollout(x0, 5, seed=11).states.copy()
    finally:
        toy_data.X[...] = original
    assert np.allclose(first, second, equal_nan=False)


def test_same_seed_reproduces_and_different_seed_can_differ(toy_model, toy_data):
    sim = NeuralSimulator(toy_model, toy_data)
    x0 = toy_data.X[int(sim.test_starts(horizon=3)[0])]
    a = sim.rollout(x0, 3, seed=7, noise_std=0.05).states
    b = sim.rollout(x0, 3, seed=7, noise_std=0.05).states
    c = sim.rollout(x0, 3, seed=8, noise_std=0.05).states
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_initial_state_perturbation_applies_only_to_x0(toy_model, toy_data):
    sim = NeuralSimulator(toy_model, toy_data)
    x0 = toy_data.X[int(sim.test_starts(horizon=3)[0])]
    spec = PerturbationSpec(
        target="AVEL", protocol="initial_state", mode="additive", amplitude=1.5
    )
    base = sim.rollout(x0, 3, seed=1)
    pert = sim.rollout(x0, 3, seed=1, perturbation=spec)
    idx = toy_data.neuron_index("AVEL")
    assert np.isclose(pert.states[0, idx] - base.states[0, idx], 1.5)
    assert pert.metadata["perturbation_notes"]["applied"] == "initial_state_only"


def test_dynamical_input_has_a_finite_window(toy_model, toy_data):
    sim = NeuralSimulator(toy_model, toy_data)
    x0 = toy_data.X[int(sim.test_starts(horizon=5)[0])]
    spec = PerturbationSpec(
        target="AVEL",
        protocol="dynamical_input",
        mode="additive",
        amplitude=2.0,
        onset=1,
        duration=2,
    )
    out = sim.rollout(x0, 5, seed=2, perturbation=spec)
    # Step 0 perturbs the state used as t+1? The callback receives s=0 after
    # predicting t+1, so onset=1 affects states 2 only in a two-step window.
    assert out.metadata["perturbation_notes"]["window"] == [1, 3]
    assert out.states.shape == (6, toy_data.n_neurons)


def test_empirical_response_is_refused_without_evidence(toy_model, toy_data):
    sim = NeuralSimulator(toy_model, toy_data)
    with pytest.raises(PermissionError, match="matched"):
        sim.rollout(
            toy_data.X[0],
            3,
            perturbation=PerturbationSpec(target="AVEL", protocol="empirical_response"),
        )


def test_unknown_target_and_label_neuron_are_rejected(toy_model, toy_data):
    sim = NeuralSimulator(toy_model, toy_data)
    with pytest.raises((KeyError, ValueError)):
        sim.rollout(
            toy_data.X[0],
            2,
            perturbation=PerturbationSpec(target="NOT_A_CELL"),
        )
    # AVAL is a Kato label neuron and absent from the toy predictor set.
    with pytest.raises((KeyError, ValueError)):
        sim.rollout(
            toy_data.X[0],
            2,
            perturbation=PerturbationSpec(target="AVAL"),
        )


def test_metrics_separate_one_step_and_rollout(toy_model, toy_data):
    sim = NeuralSimulator(toy_model, toy_data)
    one = one_step_metrics(sim)
    assert one["n_starts"] > 0
    assert np.isfinite(one["mse"])
    summary, rows = rollout_metrics(sim, horizon=5)
    assert summary["n_starts"] == len(rows)
    assert summary["horizon"] == 5
    assert 0.0 <= summary["finite_fraction"] <= 1.0
