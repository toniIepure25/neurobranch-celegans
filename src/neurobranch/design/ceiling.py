"""Design calculations built on the metric-aligned reliability model.

The binding design question in this atlas is not *which target to stimulate
next* but *how many animals must be pooled before a target profile is
measured well enough for the intended claim*.  The functions here invert the
pooled reliability formula and report the animal budget implied by a goal.

Assumptions are explicit and additive: a target profile ``theta_r`` shared
across animals, animal-specific deviations with variance ``interaction``, and
sampling noise with variance ``noise``.  Reliability of a mean over ``n``
animals is then ``signal / (signal + (noise + interaction) / n)``.  This is a
design approximation, not a mechanistic model.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_GOALS: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9)


def animals_for_reliability(
    signal: float,
    noise: float,
    interaction: float,
    target_reliability: float,
) -> float:
    """Animals needed to reach ``target_reliability``.

    Returns ``inf`` when the goal is unattainable (``target >= 1`` or
    ``signal <= 0``) and ``1.0`` when the error variance is zero.  The result
    is a continuous budget; round up for a concrete design.
    """
    if not 0.0 < target_reliability < 1.0:
        if target_reliability <= 0.0:
            return 0.0
        return math.inf
    if not np.isfinite(signal) or signal <= 0.0:
        return math.inf
    error = float(noise) + float(interaction)
    if not np.isfinite(error) or error <= 0.0:
        return 1.0
    return float(error * target_reliability / ((1.0 - target_reliability) * signal))


def required_animals_by_target(
    components: pd.DataFrame,
    *,
    goal: float,
) -> pd.Series:
    """Required animals for every target in a component table."""
    if components.empty:
        return pd.Series(dtype=float)
    required = [
        animals_for_reliability(row.signal, row.noise, row.interaction, goal)
        for row in components.itertuples(index=False)
    ]
    return pd.Series(required, index=components["target"].to_numpy(dtype=str), name=f"animals_for_{goal:g}")


def design_table(
    components: pd.DataFrame,
    *,
    goals: Iterable[float] = DEFAULT_GOALS,
) -> pd.DataFrame:
    """Summarise the animal budget for a grid of reliability goals.

    ``median_required`` and ``p90_required`` describe the target population;
    ``fraction_feasible`` is the share of targets whose budget is finite.
    """
    columns = [
        "goal",
        "n_targets",
        "median_required",
        "p90_required",
        "max_required",
        "fraction_feasible",
    ]
    if components.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, float]] = []
    for goal in goals:
        required = required_animals_by_target(components, goal=goal).to_numpy(dtype=float)
        finite = required[np.isfinite(required)]
        rows.append(
            {
                "goal": float(goal),
                "n_targets": float(required.size),
                "median_required": float(np.median(finite)) if finite.size else float("nan"),
                "p90_required": float(np.percentile(finite, 90)) if finite.size else float("nan"),
                "max_required": float(finite.max()) if finite.size else float("nan"),
                "fraction_feasible": float(finite.size / required.size) if required.size else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "DEFAULT_GOALS",
    "animals_for_reliability",
    "design_table",
    "required_animals_by_target",
]
