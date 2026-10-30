"""Posterior-predictive animal-budget calculation.

This is a planning tool, not a validated prospective design.  It propagates the
hierarchical posterior uncertainty into the probability that a target reaches a
chosen reliability at a given animal count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.benchmark.hierarchical import HierarchicalResult


def reliability_curve(
    sigma2: np.ndarray,
    tau2: np.ndarray,
    nu: float,
    *,
    max_animals: int = 60,
) -> pd.DataFrame:
    """Posterior median and 95% interval of R(n) for n = 1..max_animals."""
    if max_animals < 1:
        raise ValueError("max_animals must be >= 1")
    rows: list[dict[str, float]] = []
    for n in range(1, max_animals + 1):
        denominator = sigma2 + (tau2 + nu) / float(n)
        values = sigma2 / np.where(denominator > 0, denominator, np.nan)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "n_animals": float(n),
                "reliability_median": float(np.median(values)),
                "reliability_low": float(np.quantile(values, 0.025)),
                "reliability_high": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def design_table(
    result: HierarchicalResult,
    *,
    target_reliability: float = 0.5,
    probability: float = 0.8,
    max_animals: int = 60,
) -> pd.DataFrame:
    """Minimum animal count reaching a reliability target with posterior probability."""
    if not 0 < target_reliability < 1:
        raise ValueError("target_reliability must be in (0, 1)")
    if not 0 < probability < 1:
        raise ValueError("probability must be in (0, 1)")
    rows: list[dict[str, object]] = []
    values = result.by_target.set_index("target")
    for target, draws in result.draws_by_target.items():
        sigma2, tau2 = draws[:, 0], draws[:, 1]
        if target not in values.index:
            continue
        nu = float(values.loc[target, "mean_sampling_var"])
        required = None
        best = 0.0
        for n in range(1, max_animals + 1):
            denominator = sigma2 + (tau2 + nu) / float(n)
            reliability = sigma2 / np.where(denominator > 0, denominator, np.nan)
            probability_reached = float(np.mean(reliability >= target_reliability))
            best = max(best, probability_reached)
            if probability_reached >= probability:
                required = n
                break
        rows.append(
            {
                "target": target,
                "required_animals": required if required is not None else max_animals + 1,
                "probability_at_required": best,
                "target_reliability": target_reliability,
                "requested_probability": probability,
                "status": "ok" if required is not None else "not_reached_within_budget",
            }
        )
    return pd.DataFrame(rows)


def design_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"status": "no_targets"}
    ok = frame[frame["status"] == "ok"]
    return {
        "status": "ok",
        "n_targets": int(len(frame)),
        "n_reached": int(len(ok)),
        "median_required_animals": float(ok["required_animals"].median()) if len(ok) else None,
        "max_required_animals": int(ok["required_animals"].max()) if len(ok) else None,
        "caveat": "Planning approximation conditional on the hierarchical model; not prospectively validated.",
    }


__all__ = ["design_summary", "design_table", "reliability_curve"]
