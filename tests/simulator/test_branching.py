from __future__ import annotations

import numpy as np

from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.branching import (
    entropy,
    js,
    outcome_counts,
    reversal_events,
    reversal_states,
    simulated_local,
    tv,
)


def test_distribution_helpers():
    p = np.array([0.5, 0.5, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.isclose(tv(p, q), 0.5)
    assert entropy(p) > 0
    assert entropy(np.array([1.0, 0, 0, 0])) == 0
    assert js(p, q) > 0


def test_outcome_counts_uses_first_departure():
    labels = np.array([1, 1, 2, 2, 3, 3])
    counts = outcome_counts(labels, initial=1)
    assert counts[2] == 1
    assert counts.sum() == 1


def test_reversal_windows_are_non_overlapping(toy_data):
    events = reversal_events(toy_data, horizon=10)
    states = reversal_states(toy_data, horizon=10)
    assert np.all(np.diff(states) >= 10) if len(states) > 1 else True
    assert set(events.tolist()).issubset(set(states.tolist())) or len(states) == 0


def test_simulated_local_is_a_distribution(toy_model, toy_data):
    sim = NeuralSimulator(toy_model, toy_data)
    frame = int(np.flatnonzero(toy_data.masks["test"])[0])
    p, meta = simulated_local(sim, frame, horizon=5, n_rollouts=8, noise_std=0.01)
    assert p.shape == (4,)
    assert np.isclose(p.sum(), 1.0)
    assert meta["n_rollouts"] == 8
