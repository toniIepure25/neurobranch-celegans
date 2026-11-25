"""Trajectory estimates using separate support and passive-occupancy designs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurobranch.abstraction.ic1_v2_contract import InterventionDesign


@dataclass(frozen=True)
class IntervalEstimate:
    value: float
    lower: float
    upper: float


@dataclass(frozen=True)
class SampleResult:
    overall: IntervalEstimate
    cells: dict[int, IntervalEstimate]
    response_counts: dict[int, tuple[int, ...]]
    unobserved_passive_starts: tuple[int, ...]


def _positive_cells(design: InterventionDesign) -> dict[int, NDArray[np.intp]]:
    labels = np.asarray(design.fixture.partition)
    active = np.asarray(design.passive_weights) > 0
    return {int(label): np.flatnonzero(active & (labels == label)) for label in np.unique(labels[active])}


def _sample_response_rows(
    design: InterventionDesign, starts: NDArray[np.intp], trials: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    if type(trials) is not int or trials < 1:
        raise ValueError("trials per start must be a positive integer")
    states = np.repeat(starts, trials)
    chains = len(states)
    for operator in design.operators:
        states = design.fixture.simulate(operator, n_chains=chains, n_steps=1, rng=rng, initial=states)[:, 1]

    cumulative = np.cumsum(design.emission[states], axis=1)
    cumulative[:, -1] = 1.0
    responses = (rng.random((chains, 1)) >= cumulative).sum(axis=1)
    categories = design.emission.shape[1]
    row_ids = np.repeat(np.arange(len(starts)), trials)
    return np.bincount(row_ids * categories + responses, minlength=len(starts) * categories).reshape(-1, categories)


def _empirical_distances(counts: NDArray[np.int64], trials: int) -> NDArray[np.float64]:
    differences = counts[:, None, :] - counts[None, :, :]
    return np.abs(differences).sum(axis=-1) / (2 * trials)


def _row_bound(counts: NDArray[np.int64], trials: int, alpha: float) -> float:
    rows, categories = counts.shape
    log_terms = (categories + 1) * math.log(2) + math.log(rows / alpha)
    return min(1.0, math.sqrt(log_terms / (2 * trials)))


def _interval(value: float, radius: float) -> IntervalEstimate:
    return IntervalEstimate(value, max(0.0, value - radius), min(1.0, value + radius))


def _as_counts(starts: NDArray[np.intp], counts: NDArray[np.int64]) -> dict[int, tuple[int, ...]]:
    return {int(start): tuple(int(value) for value in row) for start, row in zip(starts, counts, strict=True)}


def estimate_support_diameter(
    design: InterventionDesign, *, trials_per_start: int, rng: np.random.Generator
) -> SampleResult:
    """Stratify over every positive passive start, including rare states."""
    cells = _positive_cells(design)
    starts = np.sort(np.concatenate(list(cells.values())))
    counts = _sample_response_rows(design, starts, trials_per_start, rng)
    distances = _empirical_distances(counts, trials_per_start)
    radius = 2 * _row_bound(counts, trials_per_start, alpha=0.05)
    estimates = {}
    for cell, members in cells.items():
        rows = np.searchsorted(starts, members)
        value = float(distances[np.ix_(rows, rows)].max())
        estimates[cell] = _interval(value, radius if len(members) > 1 else 0.0)

    overall = IntervalEstimate(
        max(estimate.value for estimate in estimates.values()),
        max(estimate.lower for estimate in estimates.values()),
        max(estimate.upper for estimate in estimates.values()),
    )
    return SampleResult(overall, estimates, _as_counts(starts, counts), ())


def estimate_occupancy_weighted(
    design: InterventionDesign, *, pairs_per_cell: int, trials_per_start: int, rng: np.random.Generator
) -> SampleResult:
    """Sample independent passive start pairs and independent conditional response trials."""
    if type(pairs_per_cell) is not int or pairs_per_cell < 1:
        raise ValueError("start pairs per cell must be a positive integer")
    cells = _positive_cells(design)
    weights = np.asarray(design.passive_weights, dtype=np.int64)

    pairs_by_cell = {}
    masses = {}
    for cell, members in cells.items():
        cumulative = np.cumsum(weights[members], dtype=np.int64)
        masses[cell] = int(cumulative[-1])
        tickets = rng.integers(masses[cell], size=(pairs_per_cell, 2))
        pairs_by_cell[cell] = members[np.searchsorted(cumulative, tickets, side="right")]
    starts = np.unique(np.concatenate(list(pairs_by_cell.values())).ravel())

    counts = _sample_response_rows(design, starts, trials_per_start, rng)
    distances = _empirical_distances(counts, trials_per_start)
    row_radius = 2 * _row_bound(counts, trials_per_start, alpha=0.025)
    pair_radius = math.sqrt(math.log(2 * len(cells) / 0.025) / (2 * pairs_per_cell))

    estimates = {}
    for cell, pairs in pairs_by_cell.items():
        rows = np.searchsorted(starts, pairs)
        value = float(distances[rows[:, 0], rows[:, 1]].mean())
        unequal = float((pairs[:, 0] != pairs[:, 1]).mean())
        radius = row_radius * unequal + pair_radius if len(cells[cell]) > 1 else 0.0
        estimates[cell] = _interval(value, radius)

    total = sum(masses.values())
    overall = IntervalEstimate(
        sum(masses[cell] / total * estimate.value for cell, estimate in estimates.items()),
        sum(masses[cell] / total * estimate.lower for cell, estimate in estimates.items()),
        sum(masses[cell] / total * estimate.upper for cell, estimate in estimates.items()),
    )
    unobserved = np.setdiff1d(np.flatnonzero(weights), starts)
    return SampleResult(overall, estimates, _as_counts(starts, counts), tuple(int(state) for state in unobserved))
