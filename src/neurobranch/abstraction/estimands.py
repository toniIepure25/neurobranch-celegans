"""Commutation-error estimands for micro-to-macro neural abstractions.

Two errors are defined per macro cell ``c`` and operator ``u``:

* ``dynamic``: how much the distribution of the next macro state depends on which
  member of ``c`` we started from (sup over pairs, total variation);
* ``behaviour``: the same for the behavioural readout.

The supremum is deliberate.  A mean over members hides exactly the aliasing that a
coarse abstraction introduces, which is the failure mode this benchmark exists to
detect.  Values are computed from conditional distributions, never from a model's
predictions, so an enumerating reference and a sampling estimator can be compared
without sharing code.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_CELL_COUNT = 30
NOMINAL_COVERAGE = 0.95
TOLERANCE = 0.05


class EstimandError(ValueError):
    """Raised when an estimand cannot be formed from the supplied arrays."""


def as_distribution(values: Sequence[float], *, name: str = "distribution", atol: float = 1e-9) -> np.ndarray:
    """Return a validated probability vector."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise EstimandError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise EstimandError(f"{name} contains non-finite values")
    if np.any(array < -atol):
        raise EstimandError(f"{name} contains negative values")
    total = float(array.sum())
    if total <= 0:
        raise EstimandError(f"{name} has zero mass")
    return array / total


def tv_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Total variation distance between two categorical distributions."""
    a = as_distribution(p, name="p")
    b = as_distribution(q, name="q")
    if a.shape != b.shape:
        raise EstimandError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(0.5 * np.abs(a - b).sum())


def wasserstein_1d(p: Sequence[float], q: Sequence[float], grid: Sequence[float]) -> float:
    """First Wasserstein distance on a supplied ordered grid (categorical masses)."""
    a = as_distribution(p, name="p")
    b = as_distribution(q, name="q")
    positions = np.asarray(grid, dtype=float)
    if a.shape != positions.shape or b.shape != positions.shape:
        raise EstimandError("masses and grid must share length")
    if positions.size < 2:
        raise EstimandError("grid must have at least two points")
    if np.any(np.diff(positions) <= 0):
        raise EstimandError("grid must be strictly increasing")
    cdf_gap = np.abs(np.cumsum(a - b))
    widths = np.diff(positions)
    return float(np.sum(cdf_gap[:-1] * widths))


def sup_pairwise_tv(rows: np.ndarray) -> tuple[float, int]:
    """Supremum total variation over all pairs of rows, plus the number of rows."""
    array = np.asarray(rows, dtype=float)
    if array.ndim != 2:
        raise EstimandError(f"rows must be 2-D, got shape {array.shape}")
    if array.shape[0] == 0:
        return 0.0, 0
    if array.shape[0] == 1:
        return 0.0, 1
    normalised = np.vstack([as_distribution(row, name=f"row[{i}]") for i, row in enumerate(array)])
    best = 0.0
    for i in range(normalised.shape[0] - 1):
        diff = 0.5 * np.abs(normalised[i + 1 :] - normalised[i]).sum(axis=1)
        best = max(best, float(diff.max()))
    return best, normalised.shape[0]


@dataclass(frozen=True)
class CommutationError:
    """One commutation measurement for one operator."""

    operator: str
    behavior: float
    dynamic: float
    worst_case: float
    n_cells: int
    n_unsupported: int
    support_coverage: float
    ci: tuple[float, float] | None = None
    status: str = "ok"
    reason: str = ""
    min_cell_count: int = MIN_CELL_COUNT
    n_cells_singleton: int = 0
    occupancy_floor: float = 0.0
    mean_pairwise: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "behavior": self.behavior,
            "dynamic": self.dynamic,
            "worst_case": self.worst_case,
            "n_cells": self.n_cells,
            "n_cells_singleton": self.n_cells_singleton,
            "n_unsupported": self.n_unsupported,
            "support_coverage": self.support_coverage,
            "ci": list(self.ci) if self.ci is not None else None,
            "status": self.status,
            "reason": self.reason,
            "min_cell_count": self.min_cell_count,
            "occupancy_floor": self.occupancy_floor,
            "mean_pairwise": self.mean_pairwise,
            "definition": (
                "sup over within-cell pairs of total variation between next-macro-state "
                "(dynamic) or behaviour (behavior) distributions"
            ),
            "not_supported": [
                "a mean over cells (it hides aliasing)",
                "a ranking across information contracts",
                "any biological or causal interpretation",
            ],
        }


def _unsupported(
    *, operator: str, reason: str, n_cells: int = 0, n_unsupported: int = 0, min_cell_count: int = 2
) -> CommutationError:
    return CommutationError(
        operator=operator,
        behavior=float("nan"),
        dynamic=float("nan"),
        worst_case=float("nan"),
        n_cells=n_cells,
        n_unsupported=n_unsupported,
        support_coverage=0.0,
        status="insufficient_support",
        reason=reason,
        min_cell_count=min_cell_count,
    )


def commutation_from_conditional(
    *,
    operator: str,
    macro_by_state: np.ndarray,
    behavior_by_state: np.ndarray,
    partition: Sequence[int],
    state_counts: Sequence[float] | None = None,
    min_cell_count: int = 2,
    occupancy_floor: float = 0.0,
    ci: tuple[float, float] | None = None,
) -> CommutationError:
    """Exact sup-based commutation error from conditional distributions.

    ``macro_by_state[s, j]`` is ``P(next macro cell = j | state s, operator)`` and
    ``behavior_by_state[s, b]`` is ``P(behaviour = b | state s)``.  ``partition[s]``
    assigns each micro state to a macro cell.

    ``min_cell_count`` counts *micro states* per cell here, so two is the
    structural minimum (a supremum needs a pair).  The sampled estimator uses
    ``MIN_CELL_COUNT`` sampled rows instead, and reports which value it applied.

    ``occupancy_floor`` restricts the supremum to pairs that both carry at least
    ``occupancy_floor`` times the largest member weight in the cell.  Without it the
    supremum would always be attained by a pair of members that the dynamics cannot
    actually visit in the same cell, and the error could never shrink when a richer
    macro state (a longer history) is used.  The floor is a declared property of the
    estimand, not a numerical convenience, and ``0.0`` means "all members".
    """
    macro = np.asarray(macro_by_state, dtype=float)
    behavior = np.asarray(behavior_by_state, dtype=float)
    blocks = np.asarray(partition, dtype=int)
    if macro.ndim != 2 or behavior.ndim != 2:
        raise EstimandError("macro_by_state and behavior_by_state must be 2-D")
    if macro.shape[0] != blocks.size or behavior.shape[0] != blocks.size:
        raise EstimandError("state axis does not match the partition")
    if not np.all(np.isfinite(macro)) or not np.all(np.isfinite(behavior)):
        raise EstimandError("conditional distributions contain non-finite values")

    weights = np.ones(blocks.size) if state_counts is None else np.asarray(state_counts, dtype=float)
    if weights.shape != blocks.shape:
        raise EstimandError("state_counts must match the number of states")
    if np.any(weights < 0):
        raise EstimandError("state_counts must be non-negative")
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(blocks.size, 1.0 / blocks.size)

    dynamic = 0.0
    behavioural = 0.0
    weighted_pairs = 0.0
    weighted_total = 0.0
    n_cells = 0
    n_unsupported = 0
    n_singleton = 0
    supported_mass = 0.0
    for cell in np.unique(blocks):
        members = np.flatnonzero(blocks == cell)
        if members.size < min_cell_count:
            n_unsupported += 1
            continue
        if occupancy_floor > 0.0:
            peak = float(weights[members].max())
            if peak <= 0.0:
                # No member is reachable under the supplied weights: the cell carries
                # no evidence, and treating its product-space members as candidates
                # would manufacture an error that no trajectory can exhibit.
                n_unsupported += 1
                continue
            members = members[weights[members] >= occupancy_floor * peak]
            if members.size < 2:
                # Only one member is reachable: the cell is a function of the state,
                # so the abstraction is exact there by construction, not by evidence.
                n_singleton += 1
                n_cells += 1
                supported_mass += float(weights[members].sum()) if members.size else 0.0
                continue
        n_cells += 1
        supported_mass += float(weights[members].sum())
        cell_dynamic, _ = sup_pairwise_tv(macro[members])
        cell_behavioral, _ = sup_pairwise_tv(behavior[members])
        dynamic = max(dynamic, cell_dynamic)
        behavioural = max(behavioural, cell_behavioral)
        cell_weights = weights[members] / weights[members].sum()
        for first in range(members.size - 1):
            for second in range(first + 1, members.size):
                pair_weight = cell_weights[first] * cell_weights[second]
                weighted_pairs += pair_weight * tv_distance(macro[members[first]], macro[members[second]])
                weighted_total += pair_weight

    if n_cells == 0:
        return _unsupported(
            operator=operator,
            reason=f"no macro cell has at least {min_cell_count} micro states",
            n_unsupported=n_unsupported,
            min_cell_count=min_cell_count,
        )

    return CommutationError(
        operator=operator,
        behavior=behavioural,
        dynamic=dynamic,
        worst_case=max(dynamic, behavioural),
        n_cells=n_cells,
        n_unsupported=n_unsupported,
        support_coverage=float(supported_mass),
        ci=ci,
        status="ok",
        min_cell_count=min_cell_count,
        n_cells_singleton=n_singleton,
        occupancy_floor=occupancy_floor,
        mean_pairwise=(weighted_pairs / weighted_total) if weighted_total > 0 else None,
    )


def bootstrap_sup_interval(
    values_by_cell: Iterable[np.ndarray],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap for a maximum of per-cell suprema.

    ``values_by_cell`` holds, for each cell, the matrix of conditional rows whose
    pairwise total variation defines that cell's suprema.  Cells are resampled
    independently, which matches the protocol's unit of uncertainty.
    """
    cells = [np.asarray(cell, dtype=float) for cell in values_by_cell]
    if not cells:
        return float("nan"), float("nan")
    generator = rng or np.random.default_rng(0)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        pick = generator.integers(0, len(cells), size=len(cells))
        worst = 0.0
        for cell_index in pick:
            value, _ = sup_pairwise_tv(cells[cell_index])
            worst = max(worst, value)
        draws[index] = worst
    lower, upper = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lower), float(upper)


def delta_commute(
    *,
    micro_pushforward: Sequence[float],
    macro_transition: Sequence[float],
) -> float:
    """``D[ tau# P_X^u , P_C^{omega(u)} ]`` for a fixed starting cell."""
    return tv_distance(micro_pushforward, macro_transition)


def identified(operator_errors: Sequence[CommutationError]) -> bool:
    """True when every operator produced a supported commutation estimate."""
    return bool(operator_errors) and all(error.status == "ok" for error in operator_errors)
