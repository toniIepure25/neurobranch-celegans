"""Estimands: distances, the sup-based error, the support guard and the floor."""

from __future__ import annotations

import numpy as np
import pytest

from neurobranch.abstraction.estimands import (
    CommutationError,
    EstimandError,
    bootstrap_sup_interval,
    commutation_from_conditional,
    sup_pairwise_tv,
    tv_distance,
    wasserstein_1d,
)


def test_tv_distance_bounds() -> None:
    assert tv_distance([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert tv_distance([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert tv_distance([0.7, 0.3], [0.3, 0.7]) == pytest.approx(0.4)


def test_tv_distance_renormalises_and_rejects_invalid_input() -> None:
    # Unnormalised input is renormalised rather than refused, so a caller cannot
    # smuggle a weight vector in as a distribution by accident.
    assert tv_distance([0.5, 0.6], [0.5, 0.5]) == pytest.approx(0.0454545, abs=1e-6)
    with pytest.raises(EstimandError):
        tv_distance([1.0, -0.5], [0.5, 0.5])
    with pytest.raises(EstimandError):
        tv_distance([0.0, 0.0], [0.5, 0.5])
    with pytest.raises(EstimandError):
        tv_distance([1.0, 0.0], [1.0, 0.0, 0.0])


def test_wasserstein_uses_uneven_grid_widths() -> None:
    grid = [0.0, 1.0, 4.0]
    assert wasserstein_1d([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], grid) == pytest.approx(4.0)
    assert wasserstein_1d([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], grid) == pytest.approx(3.0)
    with pytest.raises(EstimandError):
        wasserstein_1d([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0])


def test_sup_pairwise_tv_takes_the_worst_pair() -> None:
    rows = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    value, count = sup_pairwise_tv(rows)
    assert value == pytest.approx(1.0)
    assert count == 3
    assert sup_pairwise_tv(rows[:1]) == (0.0, 1)


def test_zero_error_for_identical_rows() -> None:
    projection = np.array([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [1.0, 0.0]])
    behavior = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    error = commutation_from_conditional(
        operator="u",
        macro_by_state=projection,
        behavior_by_state=behavior,
        partition=[0, 0, 1, 1],
    )
    assert error.status == "ok"
    assert error.dynamic == 0.0
    assert error.n_cells_singleton == 0


def test_behaviour_and_dynamic_errors_are_reported_separately() -> None:
    projection = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    behavior = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    error = commutation_from_conditional(
        operator="u",
        macro_by_state=projection,
        behavior_by_state=behavior,
        partition=[0, 0, 1, 1],
    )
    assert error.dynamic == pytest.approx(1.0)
    assert error.behavior == 0.0
    assert error.worst_case == pytest.approx(1.0)


def test_support_guard_never_imputes() -> None:
    projection = np.array([[1.0, 0.0], [0.0, 1.0]])
    behavior = np.array([[1.0], [1.0]])
    error = commutation_from_conditional(
        operator="u",
        macro_by_state=projection,
        behavior_by_state=behavior,
        partition=[0, 1],
        min_cell_count=2,
    )
    assert error.status == "insufficient_support"
    assert np.isnan(error.dynamic)
    assert error.n_unsupported == 2


def test_occupancy_floor_excludes_unreachable_members() -> None:
    projection = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    behavior = np.array([[1.0], [1.0], [1.0]])
    weights = np.array([0.5, 0.0, 0.5])
    without_floor = commutation_from_conditional(
        operator="u",
        macro_by_state=projection,
        behavior_by_state=behavior,
        partition=[0, 0, 0],
        state_counts=weights,
        occupancy_floor=0.0,
    )
    with_floor = commutation_from_conditional(
        operator="u",
        macro_by_state=projection,
        behavior_by_state=behavior,
        partition=[0, 0, 0],
        state_counts=weights,
        occupancy_floor=1e-6,
    )
    assert without_floor.dynamic == pytest.approx(1.0)
    assert with_floor.dynamic == 0.0


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    cells = [rng.dirichlet([2.0, 2.0], size=3) for _ in range(4)]
    lower, upper = bootstrap_sup_interval(cells, n_boot=200, rng=np.random.default_rng(1))
    assert 0.0 <= lower <= upper <= 1.0


def test_error_serialises_without_losing_status() -> None:
    error = CommutationError(
        operator="u", behavior=0.1, dynamic=0.2, worst_case=0.2, n_cells=2, n_unsupported=0, support_coverage=1.0
    )
    payload = error.as_dict()
    assert payload["status"] == "ok"
    assert payload["operator"] == "u"
    assert payload["mean_pairwise"] is None
    assert "not_supported" in payload
