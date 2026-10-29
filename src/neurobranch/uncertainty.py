"""Two-way (animal x responder) bootstrap for profile reliability.

The animal jackknife in :mod:`neurobranch.benchmark.robustness` resamples only
animals and therefore ignores uncertainty in the responder sample.  This module
resamples both, using the same moment estimator as the point estimate.

The interval is a percentile bootstrap.  On the real atlas, resampling animal
rows with replacement duplicates them and biases the interaction variance
downward, so the interval can exclude its own point estimate.  The summary
therefore exposes ``interval_status``; a failed status means the interval must
not be reported as calibrated.  The synthetic study in
:mod:`neurobranch.design.calibration` is necessary but not sufficient evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.benchmark.variance import (
    DEFAULT_ANIMAL_COLUMN,
    _components_from_arrays,
    cell_summaries,
)


@dataclass(frozen=True)
class BootstrapResult:
    """Per-target and aggregate intervals from a two-way bootstrap."""

    by_target: pd.DataFrame
    summary: dict[str, Any]
    draws: np.ndarray


def _matrices(
    cells: pd.DataFrame, *, min_events: int, min_responder_coverage: float = 0.0
) -> tuple[np.ndarray, np.ndarray] | None:
    subset = cells[cells["n"] >= min_events]
    if subset.empty:
        return None
    means = subset.pivot_table(index="_animal", columns="_neuron", values="mean", aggfunc="mean")
    sampling_cells = subset.assign(se2=subset["var"] / np.maximum(subset["n"], 1.0))
    sampling = sampling_cells.pivot_table(index="_animal", columns="_neuron", values="se2", aggfunc="mean")
    means = means.dropna(axis=1, how="all")
    sampling = sampling.reindex(columns=means.columns)
    if min_responder_coverage > 0:
        coverage = means.notna().sum(axis=0) / max(means.shape[0], 1)
        keep = coverage >= min_responder_coverage
        means = means.loc[:, keep]
        sampling = sampling.loc[:, keep]
    if means.shape[1] < 3:
        return None
    return means.to_numpy(dtype=float), sampling.to_numpy(dtype=float)


def _percentile_interval(values: np.ndarray, level: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size < 10:
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high)


def two_way_bootstrap(
    trials: pd.DataFrame,
    *,
    fold_size: int = 22,
    n_boot: int = 500,
    level: float = 0.95,
    seed: int = 11,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
    min_responder_coverage: float = 0.0,
) -> BootstrapResult:
    """Percentile bootstrap resampling animals and responders within a target.

    Args:
        trials: event-level trial table with one row per animal/target/responder
            event.
        fold_size: animal count used for the pooled reliability estimand.
        n_boot: bootstrap replicates per target.
        level: nominal interval level.
        seed: seed for the animal and responder indices.
        response: response column.
        animal: animal identifier column.
        min_events: minimum events per cell.
        min_animals: minimum distinct animals for a target to be eligible.
        min_responder_coverage: minimum fraction of animals in which a responder
            must be observed.  The default 0.0 reproduces the Phase 10
            estimator; Phase 11 uses 0.8 to avoid sparse-coverage artefacts.

    Returns:
        A :class:`BootstrapResult` with one row per target, an aggregate summary
        and the matrix of aggregate draws.
    """
    if n_boot < 20:
        raise ValueError("n_boot must be at least 20")
    cells = cell_summaries(trials, response=response, animal=animal)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    target_draws: list[np.ndarray] = []
    single_draws: list[np.ndarray] = []
    n_failures = 0

    for target, tgroup in cells.groupby("_target", sort=False):
        matrices = _matrices(
            tgroup, min_events=min_events, min_responder_coverage=min_responder_coverage
        )
        if matrices is None:
            continue
        values, noise = matrices
        point = _components_from_arrays(values, noise, min_animals=min_animals)
        if point is None:
            continue
        n_animals = values.shape[0]
        n_responders = values.shape[1]
        pooled_draws = np.full(n_boot, np.nan)
        one_draws = np.full(n_boot, np.nan)
        for b in range(n_boot):
            animal_index = rng.integers(0, n_animals, size=n_animals)
            responder_index = rng.integers(0, n_responders, size=n_responders)
            resampled = _components_from_arrays(
                values[np.ix_(animal_index, responder_index)],
                noise[np.ix_(animal_index, responder_index)],
                min_animals=min_animals,
            )
            if resampled is None:
                n_failures += 1
                continue
            pooled_draws[b] = resampled.reliability_at(fold_size)
            one_draws[b] = resampled.single_animal_reliability
        pooled_low, pooled_high = _percentile_interval(pooled_draws, level)
        one_low, one_high = _percentile_interval(one_draws, level)
        rows.append(
            {
                "target": target,
                "n_animals": n_animals,
                "n_responders": n_responders,
                "point_pooled_reliability": point.reliability_at(fold_size),
                "pooled_ci_low": pooled_low,
                "pooled_ci_high": pooled_high,
                "point_single_animal_reliability": point.single_animal_reliability,
                "single_ci_low": one_low,
                "single_ci_high": one_high,
                "bootstrap_se_pooled": float(np.nanstd(pooled_draws, ddof=1)),
                "n_bootstrap_valid": int(np.isfinite(pooled_draws).sum()),
            }
        )
        target_draws.append(pooled_draws)
        single_draws.append(one_draws)

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "target",
                "n_animals",
                "n_responders",
                "point_pooled_reliability",
                "pooled_ci_low",
                "pooled_ci_high",
                "point_single_animal_reliability",
                "single_ci_low",
                "single_ci_high",
                "bootstrap_se_pooled",
                "n_bootstrap_valid",
            ]
        )
        return BootstrapResult(
            by_target=empty,
            summary={
                "n_targets": 0,
                "n_boot": n_boot,
                "fold_size": fold_size,
                "level": level,
                "seed": seed,
                "method": "two-way percentile bootstrap (animals x responders)",
                "n_bootstrap_failures": n_failures,
            },
            draws=np.empty(0),
        )

    draw_matrix = np.vstack(target_draws)
    aggregate = np.nanmean(draw_matrix, axis=0)
    pooled_low, pooled_high = _percentile_interval(aggregate, level)
    single_matrix = np.vstack(single_draws)
    single_aggregate = np.nanmean(single_matrix, axis=0)
    single_low, single_high = _percentile_interval(single_aggregate, level)
    by_target = pd.DataFrame(rows).sort_values("point_pooled_reliability", ascending=False).reset_index(drop=True)
    bootstrap_mean = float(np.nanmean(aggregate))
    point_mean = float(by_target["point_pooled_reliability"].mean())
    interval_status = (
        "ok" if pooled_low <= point_mean <= pooled_high else "failed_point_outside_interval"
    )
    summary = {
        "n_targets": int(len(by_target)),
        "n_boot": n_boot,
        "fold_size": fold_size,
        "level": level,
        "seed": seed,
        "method": "two-way percentile bootstrap (animals x responders)",
        "mean_pooled_reliability": float(by_target["point_pooled_reliability"].mean()),
        "pooled_ci_low": pooled_low,
        "pooled_ci_high": pooled_high,
        "mean_single_animal_reliability": float(by_target["point_single_animal_reliability"].mean()),
        "single_ci_low": single_low,
        "single_ci_high": single_high,
        "n_bootstrap_failures": n_failures,
        "bootstrap_mean": bootstrap_mean,
        "bias_vs_point": bootstrap_mean - point_mean,
        "point_estimate_in_interval": interval_status == "ok",
        "interval_status": interval_status,
        "min_responder_coverage": min_responder_coverage,
        "mean_bootstrap_se_pooled": float(by_target["bootstrap_se_pooled"].mean()),
    }
    return BootstrapResult(by_target=by_target, summary=summary, draws=aggregate)


def components_at_point(
    trials: pd.DataFrame,
    *,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
) -> pd.DataFrame:
    """Point components for callers that only need the estimator output."""
    from neurobranch.benchmark.variance import profile_components_by_target

    return profile_components_by_target(
        trials,
        response=response,
        animal=animal,
        min_events=min_events,
        min_animals=min_animals,
    )


__all__ = ["BootstrapResult", "components_at_point", "two_way_bootstrap"]
