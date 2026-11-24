"""Ground-truth fixtures: structure, declared signatures and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from neurobranch.abstraction.reference import exact_commutation, is_exactly_lumpable
from neurobranch.abstraction.systems import HISTORY_LENGTHS, RateCircuit, discrete_fixtures


@pytest.mark.parametrize("name", sorted(discrete_fixtures()))
def test_every_fixture_is_a_valid_chain(name: str) -> None:
    fixture = discrete_fixtures()[name]
    assert fixture.partition.shape == (fixture.n_states,)
    assert fixture.behavior.shape == (fixture.n_states, fixture.n_behaviors)
    for operator in fixture.operators:
        kernel = fixture.kernel(operator)
        assert kernel.shape == (fixture.n_states, fixture.n_states)
        assert np.allclose(kernel.sum(axis=1), 1.0, atol=1e-9)
        assert kernel.min() >= -1e-12


def test_positive_control_is_exactly_lumpable() -> None:
    fixture = discrete_fixtures()["exactly_lumpable/positive"]
    assert is_exactly_lumpable(fixture, "passive")
    error = exact_commutation(fixture)["passive"]
    assert error.dynamic == 0.0
    assert error.behavior == 0.0
    assert error.n_cells_singleton == 0


def test_behaviour_only_negative_breaks_dynamics_and_not_behaviour() -> None:
    fixture = discrete_fixtures()["exactly_lumpable/behaviour_only"]
    assert not is_exactly_lumpable(fixture, "passive")
    error = exact_commutation(fixture)["passive"]
    assert error.dynamic == pytest.approx(0.18, abs=1e-9)
    assert error.behavior == 0.0


def test_intervention_only_negative_is_hidden_from_the_passive_operator() -> None:
    fixture = discrete_fixtures()["exactly_lumpable/intervention_only"]
    errors = exact_commutation(fixture)
    assert errors["passive"].dynamic == 0.0
    assert errors["lesion"].dynamic == pytest.approx(0.5, abs=1e-9)


def test_history_ladder_goes_from_one_to_zero() -> None:
    one = exact_commutation(discrete_fixtures()["ring/L1"])["passive"]
    two = exact_commutation(discrete_fixtures()["ring/L2"])["passive"]
    assert one.dynamic > 0.05
    assert two.dynamic == 0.0
    assert two.n_cells_singleton == two.n_cells


def test_witness_does_not_shrink_with_history() -> None:
    values = []
    for length in HISTORY_LENGTHS[1:]:
        error = exact_commutation(discrete_fixtures()[f"ring_witness/L{length}"])["asym"]
        values.append(error.dynamic)
    assert all(value > 0.05 for value in values)
    assert max(values) - min(values) < 1e-12


def test_witness_is_benign_under_the_passive_operator() -> None:
    error = exact_commutation(discrete_fixtures()["ring_witness/L2"])["passive"]
    assert error.dynamic == 0.0


def test_stationary_weights_are_a_distribution() -> None:
    fixture = discrete_fixtures()["ring/L2"]
    weights = fixture.stationary_weights("passive")
    assert weights.shape == (fixture.n_states,)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0.0


def test_simulation_is_reproducible_for_a_fixed_seed() -> None:
    fixture = discrete_fixtures()["ring/L1"]
    first = fixture.simulate("passive", n_chains=16, n_steps=20, rng=np.random.default_rng(7))
    second = fixture.simulate("passive", n_chains=16, n_steps=20, rng=np.random.default_rng(7))
    assert np.array_equal(first, second)
    other = fixture.simulate("passive", n_chains=16, n_steps=20, rng=np.random.default_rng(8))
    assert not np.array_equal(first, other)


def test_rate_circuit_is_deterministic_and_bounded() -> None:
    circuit = RateCircuit()
    generator = np.random.default_rng(0)
    states = circuit.sample_initial(32, generator)
    stepped = circuit.step(states, "lesion", np.random.default_rng(0))
    repeated = circuit.step(states, "lesion", np.random.default_rng(0))
    assert np.allclose(stepped, repeated)
    assert np.isfinite(stepped).all()
    cells = circuit.cell_of(circuit.observe(stepped, np.random.default_rng(0)))
    assert cells.min() >= 0
    assert cells.max() < circuit.n_macro


def test_rate_circuit_rejects_unknown_operator() -> None:
    from neurobranch.abstraction.estimands import EstimandError

    circuit = RateCircuit()
    states = circuit.sample_initial(4, np.random.default_rng(0))
    with pytest.raises(EstimandError, match="unknown operator"):
        circuit.step(states, "tickle", np.random.default_rng(0))
