"""Sampling estimators for the commutation estimands.

Two entry points, both producing the same ``CommutationError`` object as the
enumerating reference:

``estimate_from_simulation``
    Dispatching a fixture through its own kernel and re-estimating the conditional
    distributions from the resulting counts.  This is the path checked against
    ``reference.exact_commutation`` in gate G1: if the simulator and the enumerator
    disagree, one of them is wrong and the benchmark is meaningless.

``estimate_from_rollout``
    A genuinely non-enumerable system (:class:`~neurobranch.abstraction.systems.RateCircuit`).
    Cells are defined on observations only, so the estimator never sees a micro state
    label; two starting states that fall in the same observed cell are compared
    directly under common random numbers.

Monte Carlo noise biases a worst-case estimator upward, because the supremum of noisy
distributions exceeds the supremum of their expectations.  Both estimators therefore
report the noise floor they can resolve and the number of replicates behind each cell,
and the rate-circuit numbers are descriptive only: no protocol gate depends on them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neurobranch.abstraction.estimands import (
    MIN_CELL_COUNT,
    CommutationError,
    EstimandError,
    commutation_from_conditional,
)
from neurobranch.abstraction.systems import DiscreteFixture, RateCircuit

DEFAULT_N_CHAINS = 400
DEFAULT_N_STEPS = 240
DEFAULT_BURN_IN = 20
DEFAULT_REPLICATES = 200


def _count_rows(
    row_counts: np.ndarray,
    *,
    n_columns: int,
    pseudo_count: float,
) -> np.ndarray:
    """Row-normalise counts with add-``pseudo_count`` smoothing."""
    smoothed = np.asarray(row_counts, dtype=float) + pseudo_count
    totals = smoothed.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    out = smoothed / totals
    if out.shape[1] != n_columns:
        raise EstimandError("count matrix has the wrong number of columns")
    return out


def estimate_from_simulation(
    fixture: DiscreteFixture,
    operator: str,
    *,
    n_chains: int = DEFAULT_N_CHAINS,
    n_steps: int = DEFAULT_N_STEPS,
    burn_in: int = DEFAULT_BURN_IN,
    rng: np.random.Generator | None = None,
    min_cell_count: int = MIN_CELL_COUNT,
    pseudo_count: float = 0.5,
    occupancy_floor: float = 1e-6,
) -> tuple[CommutationError, dict[str, Any]]:
    """Estimate the commutation error from trajectories of a discrete fixture."""
    generator = rng or np.random.default_rng(0)
    states = fixture.simulate(
        operator, n_chains=n_chains, n_steps=n_steps + burn_in, rng=generator
    )[:, burn_in:]
    current = states[:, :-1].ravel()
    following = fixture.partition[states[:, 1:].ravel()]
    n_macro = fixture.n_cells

    macro_counts = np.zeros((fixture.n_states, n_macro))
    np.add.at(macro_counts, (current, following), 1.0)

    behaviours = generator.random(current.shape) < np.cumsum(fixture.behavior, axis=1)[current, 0]
    behaviour_counts = np.zeros((fixture.n_states, fixture.n_behaviors))
    np.add.at(behaviour_counts, (current, behaviours.astype(int)), 1.0)

    visits = np.zeros(fixture.n_states)
    np.add.at(visits, current, 1.0)

    cell_visits = np.zeros(n_macro)
    for cell in range(n_macro):
        cell_visits[cell] = visits[fixture.partition == cell].sum()
    weak = {cell for cell in range(n_macro) if cell_visits[cell] < min_cell_count}
    weights = visits.copy()
    for cell in weak:
        weights[fixture.partition == cell] = 0.0
    if weights.sum() > 0:
        weights = weights / weights.sum()

    error = commutation_from_conditional(
        operator=operator,
        macro_by_state=_count_rows(macro_counts, n_columns=n_macro, pseudo_count=pseudo_count),
        behavior_by_state=_count_rows(
            behaviour_counts, n_columns=fixture.n_behaviors, pseudo_count=pseudo_count
        ),
        partition=fixture.partition,
        state_counts=weights,
        min_cell_count=2,
        occupancy_floor=occupancy_floor,
    )
    diagnostics = {
        "estimator": "simulation",
        "n_chains": n_chains,
        "n_steps": n_steps,
        "burn_in": burn_in,
        "pseudo_count": pseudo_count,
        "cell_visits": {int(cell): int(cell_visits[cell]) for cell in range(n_macro)},
        "weak_cells": sorted(int(cell) for cell in weak),
        "visits_without_support": int(sum(cell_visits[cell] for cell in weak)),
        "noise_floor": float(np.sqrt(2.0 * pseudo_count / max(n_chains, 1))),
    }
    return error, diagnostics


def estimate_from_rollout(
    circuit: RateCircuit,
    operator: str,
    *,
    n_chains: int = DEFAULT_N_CHAINS,
    n_steps: int = DEFAULT_N_STEPS,
    burn_in: int = DEFAULT_BURN_IN,
    replicates: int = DEFAULT_REPLICATES,
    rng: np.random.Generator | None = None,
    min_cell_count: int = MIN_CELL_COUNT,
    noise: float = 0.08,
) -> tuple[CommutationError, dict[str, Any]]:
    """Estimate the commutation error for a system that cannot be enumerated.

    Cells are defined on observations.  For every observed cell, up to ``replicates``
    distinct start states are pushed one step forward under the same noise draws
    (common random numbers), and the supremum pairwise total variation between their
    next-cell distributions is the cell's error.
    """
    generator = rng or np.random.default_rng(0)
    states = circuit.sample_initial(n_chains, generator)
    for _ in range(burn_in):
        states = circuit.step(states, "passive", generator)
    cells = circuit.cell_of(circuit.observe(states, generator))
    n_macro = circuit.n_macro

    noise_draws = generator.normal(0.0, circuit.noise, size=(1, replicates, 3))
    scaled = noise_draws
    behaviour_draws = generator.random((1, replicates))

    rows: dict[int, list[np.ndarray]] = {}
    behaviour_rows: dict[int, list[np.ndarray]] = {}
    for index, state in enumerate(states):
        cell = int(cells[index])
        rows.setdefault(cell, [])
        behaviour_rows.setdefault(cell, [])
        if len(rows[cell]) >= replicates:
            continue
        start = np.repeat(state[None, :], replicates, axis=0)
        stepped = circuit.step(start, operator, _FixedNoise(scaled[0]))
        next_cells = circuit.cell_of(circuit.observe(stepped, _ZeroNoise()))
        counts = np.zeros(n_macro)
        np.add.at(counts, next_cells, 1.0)
        rows[cell].append(counts / counts.sum())
        probabilities = circuit.behavior_probs(stepped)
        choice = behaviour_draws[0] < probabilities[:, 0]
        cell_behaviour = np.zeros(probabilities.shape[1])
        np.add.at(cell_behaviour, choice.astype(int), 1.0)
        behaviour_rows[cell].append(cell_behaviour / cell_behaviour.sum())

    macro_by_state = []
    behavior_by_state = []
    partition = []
    weights = []
    cell_visits = {}
    total_visits = float(sum((cells == cell).sum() for cell in rows))
    for cell, cell_rows in sorted(rows.items()):
        cell_visits[cell] = int((cells == cell).sum())
        share = cell_visits[cell] / total_visits if total_visits > 0 else 0.0
        for row, behaviour_row in zip(cell_rows, behaviour_rows[cell]):
            macro_by_state.append(row)
            behavior_by_state.append(behaviour_row)
            partition.append(cell)
            weights.append(share / max(len(cell_rows), 1))

    if not macro_by_state:
        raise EstimandError("no observed cell received a start state")

    error = commutation_from_conditional(
        operator=operator,
        macro_by_state=np.vstack(macro_by_state),
        behavior_by_state=np.vstack(behavior_by_state),
        partition=np.asarray(partition, dtype=int),
        state_counts=np.asarray(weights, dtype=float),
        min_cell_count=2,
        occupancy_floor=0.0,
    )
    diagnostics = {
        "estimator": "rollout",
        "n_chains": n_chains,
        "burn_in": burn_in,
        "replicates": replicates,
        "cell_visits": cell_visits,
        "weak_cells": sorted(int(cell) for cell, visits in cell_visits.items() if visits < min_cell_count),
        "noise_floor": float(noise),
        "caveat": (
            "Monte Carlo noise inflates a worst-case estimator; treat these values as "
            "descriptive and do not gate any claim on them."
        ),
    }
    return error, diagnostics


class _FixedNoise:
    """Deterministic noise source used to hold perturbations fixed across start states."""

    def __init__(self, draws: np.ndarray) -> None:
        self._draws = np.asarray(draws, dtype=float)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Any = None) -> np.ndarray:
        expected = self._draws.shape if size is None else tuple(np.atleast_1d(size))
        if tuple(expected) != self._draws.shape:
            raise EstimandError(
                f"fixed noise was drawn for {self._draws.shape}, requested {tuple(expected)}"
            )
        return loc + scale * self._draws


class _ZeroNoise:
    """Noise-free observation, so the cell label reflects the state and not the draw."""

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Any = None) -> np.ndarray:
        expected = tuple(np.atleast_1d(size)) if size is not None else (1,)
        return np.full(expected, float(loc))
