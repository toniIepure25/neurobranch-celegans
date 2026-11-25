"""Acceptance controls for the versioned finite-synthetic instrument."""

from dataclasses import replace

import numpy as np
import pytest

from neurobranch.abstraction.commutation import estimate_from_simulation
from neurobranch.abstraction.ic1_v2_contract import protocol_fingerprint
from neurobranch.abstraction.ic1_v2_oracle import enumerate_response_laws
from neurobranch.abstraction.ic1_v2_sampling import estimate_occupancy_weighted, estimate_support_diameter
from neurobranch.abstraction.reference import exact_commutation
from neurobranch.abstraction.systems import DiscreteFixture, discrete_fixtures
from scripts.ic1_v2_acceptance import PINNED_PROTOCOL, SPEC, calibration_cases

CASES = {case.name: case for case in calibration_cases()}


@pytest.mark.parametrize("name", CASES)
def test_analytic_targets(name):
    case = CASES[name]
    oracle = enumerate_response_laws(case.design)
    assert oracle.diameter == pytest.approx(case.expected_diameter, abs=1e-12)
    assert oracle.weighted == pytest.approx(case.expected_weighted, abs=1e-12)


def test_historical_defect_and_repair():
    design = CASES["four_state"].design
    v1_exact = exact_commutation(design.fixture)["probe"]
    v1_sample, details = estimate_from_simulation(
        design.fixture, "probe", n_chains=40, n_steps=50, burn_in=20, rng=np.random.default_rng(0)
    )
    v2 = estimate_support_diameter(design, trials_per_start=64, rng=np.random.default_rng(1))
    assert (v1_exact.dynamic, v1_sample.dynamic, v2.overall.value) == (1, 0, 1)
    assert details["cell_visits"] == {0: 0, 1: 2000}
    assert v2.response_counts == {0: (0, 64), 1: (64, 0)}


def test_rare_members_and_zero_mass_are_different():
    design = CASES["rare"].design
    oracle = enumerate_response_laws(design)
    support = estimate_support_diameter(design, trials_per_start=16, rng=np.random.default_rng(2))
    occupancy = estimate_occupancy_weighted(
        design, pairs_per_cell=128, trials_per_start=16, rng=np.random.default_rng(3)
    )
    assert oracle.cells[0].starts == (0, 1)
    assert oracle.weighted == pytest.approx(2 * (1 - 1e-8) * 1e-8, abs=1e-20)
    assert support.response_counts.keys() == {0, 1}
    assert support.overall.value == 1
    assert occupancy.unobserved_passive_starts == (1,)
    assert occupancy.overall.value == 0
    assert occupancy.overall.lower <= oracle.weighted <= occupancy.overall.upper

    impossible = CASES["impossible"].design
    result = estimate_support_diameter(impossible, trials_per_start=16, rng=np.random.default_rng(4))
    assert result.response_counts.keys() == {0, 1}
    assert impossible.fixture.partition[2] == 0
    assert result.overall.value == 0


def test_zero_mass_bridge_is_reached_and_retained(monkeypatch):
    case = CASES["multi_step_exit"]
    observed_paths = []
    simulate = DiscreteFixture.simulate

    def record_step(fixture, operator, **kwargs):
        initial = kwargs["initial"]
        observed_paths.append((operator, set(initial)))
        return simulate(fixture, operator, **kwargs)

    monkeypatch.setattr(DiscreteFixture, "simulate", record_step)
    estimate = estimate_support_diameter(case.design, trials_per_start=32, rng=np.random.default_rng(5))
    assert observed_paths == [("probe", {0, 1}), ("follow", {1, 2})]
    assert case.design.passive_weights[2] == case.design.passive_weights[3] == 0
    assert estimate.response_counts == {0: (0, 32), 1: (32, 0)}
    assert enumerate_response_laws(case.design).response_by_start == {0: (0.0, 1.0), 1: (1.0, 0.0)}

    repeated = estimate_support_diameter(CASES["persistent"].design, trials_per_start=32,
                                         rng=np.random.default_rng(6))
    assert repeated.response_counts == {0: (32, 0), 1: (32, 0)}


def test_post_intervention_only_category_keeps_its_mass():
    design = CASES["post_only"].design
    oracle = enumerate_response_laws(design)
    result = estimate_support_diameter(design, trials_per_start=32, rng=np.random.default_rng(7))
    assert set(oracle.cells) == {0}
    assert oracle.cells[0].response_mixture == (0.5, 0.5)
    assert result.response_counts == {0: (0, 32), 1: (32, 0)}


def test_oracle_does_not_simulate_and_a_bad_simulator_is_detectable(monkeypatch):
    design = CASES["four_state"].design

    def forbid_simulation(*args, **kwargs):
        pytest.fail("the oracle entered the simulation path")

    monkeypatch.setattr(DiscreteFixture, "simulate", forbid_simulation)
    assert enumerate_response_laws(design).diameter == 1

    def erase_after_intervention(fixture, operator, *, n_chains, n_steps, rng, initial):
        return np.column_stack((initial, np.zeros(n_chains, dtype=int)))

    monkeypatch.setattr(DiscreteFixture, "simulate", erase_after_intervention)
    assert estimate_support_diameter(design, trials_per_start=16, rng=np.random.default_rng(8)).overall.value == 0
    assert enumerate_response_laws(design).diameter == 1


def test_three_category_emission_order_and_repeatability():
    design = CASES["coincident_multicategory"].design
    first = estimate_support_diameter(design, trials_per_start=8192, rng=np.random.default_rng(11))
    second = estimate_support_diameter(design, trials_per_start=8192, rng=np.random.default_rng(11))
    assert first == second
    np.testing.assert_allclose(np.array(list(first.response_counts.values())) / 8192, design.emission, atol=.02)
    assert first.overall.lower <= 0.5 <= first.overall.upper


def test_weighted_aggregation_uses_cell_and_conditional_masses():
    original = CASES["impossible"].design
    fixture = replace(original.fixture, partition=np.array([7, 7, 19]))
    design = replace(original, fixture=fixture, passive_weights=(1, 3, 4), emission=np.eye(2)[[0, 1, 1]])
    oracle = enumerate_response_laws(design)
    assert oracle.cells[7].passive_mass == 0.5
    assert oracle.cells[7].weighted == 0.375
    assert oracle.cells[19].weighted == 0
    assert oracle.diameter == 1
    assert oracle.weighted == 0.1875
    estimate = estimate_occupancy_weighted(
        design, pairs_per_cell=4096, trials_per_start=16, rng=np.random.default_rng(12)
    )
    assert estimate.overall.value == pytest.approx(oracle.weighted, abs=.02)
    assert estimate.cells[19].upper == 0


def test_ring_compatibility_and_identification_bound():
    historical = discrete_fixtures()
    for case, key, operator in (("ring_l1", "ring/L1", "passive"), ("ring_l2", "ring/L2", "passive"),
                                ("ring_asym", "ring_witness/L2", "asym")):
        result = enumerate_response_laws(CASES[case].design)
        assert result.diameter == pytest.approx(exact_commutation(historical[key])[operator].dynamic)

    design = CASES["ring_asym"].design
    passive = design.fixture.kernel("passive")
    context0, context1 = slice(0, None, 2), slice(1, None, 2)
    np.testing.assert_array_equal(passive[context0, context0], passive[context1, context1])
    assert not passive[context0, context1].any() and not passive[context1, context0].any()
    np.testing.assert_array_equal(design.emission[context0], design.emission[context1])
    assert design.passive_weights[0::2] == design.passive_weights[1::2]
    oracle = enumerate_response_laws(design)
    for cell in oracle.cells.values():
        assert cell.diameter == pytest.approx(0.85)
        for start in cell.starts:
            radius = np.abs(np.array(oracle.response_by_start[start]) - cell.response_mixture).sum() / 2
            assert radius == pytest.approx(0.425)


def test_fingerprint_preserves_content_and_normalizes_only_newlines():
    lf = SPEC.read_bytes().replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert lf != crlf
    assert protocol_fingerprint(lf) == protocol_fingerprint(crlf) == PINNED_PROTOCOL
    assert protocol_fingerprint(lf.replace(b"\n", b"\r")) == PINNED_PROTOCOL
    assert protocol_fingerprint(lf.replace(b"8192", b"4096")) != PINNED_PROTOCOL


@pytest.mark.parametrize("changes", [
    {"passive_weights": (0, 0, 0, 0)},
    {"passive_weights": (1, -1, 0, 0)},
    {"operators": ()},
    {"operators": ("unknown",)},
    {"emission": np.zeros((4, 3))},
])
def test_invalid_contract_rejected(changes):
    with pytest.raises(ValueError):
        replace(CASES["four_state"].design, **changes)


def test_negative_transition_rejected_even_with_nearly_unit_row_sum():
    design = CASES["four_state"].design
    kernel = design.fixture.kernel("probe").copy()
    kernel[0, :3] = [-1e-13, 0.0, 1.0 + 1e-13]
    fixture = replace(design.fixture, kernels={**design.fixture.kernels, "probe": kernel})
    with pytest.raises(ValueError, match="stochastic rows"):
        replace(design, fixture=fixture)
