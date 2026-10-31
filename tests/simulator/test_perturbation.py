from __future__ import annotations

import numpy as np

from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.perturbation import (
    CANDIDATES,
    baseline_for,
    candidate_specs,
    effect_row,
    initial_state_specs,
    outcome_distribution,
)
from neurobranch.simulator.types import WARNING


def test_candidate_and_initial_specs_are_explicit():
    assert len(CANDIDATES) == 6
    assert len(candidate_specs()) == 12
    assert len(initial_state_specs()) == 6
    assert {s.protocol for s in initial_state_specs()} == {"initial_state"}


def test_outcome_distribution_sums_to_one():
    class R:
        behavior_proba = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])

    p = outcome_distribution([R(), R()], initial=0)
    assert np.isclose(p.sum(), 1.0)
    assert p[1] == 1.0


def test_effect_row_is_model_dependent_and_finite(toy_model, toy_data):
    sim = NeuralSimulator(toy_model, toy_data)
    frame = int(sim.test_starts(horizon=4)[0])
    base = baseline_for(sim, frame, horizon=4, n_rollouts=4, noise_std=0.0, base_seed=1)
    from neurobranch.simulator.types import PerturbationSpec

    spec = PerturbationSpec(target="AVEL", amplitude=1.0, onset=0, duration=3)
    row = effect_row(
        sim,
        frame,
        spec,
        base,
        horizon=4,
        n_rollouts=4,
        noise_std=0.0,
        base_seed=1,
    )
    assert row["target"] == "AVEL"
    assert np.isfinite(row["delta_tv"])
    assert row["delta_tv"] >= 0
    assert row["warning"] == WARNING
    assert row["finite"] is True
