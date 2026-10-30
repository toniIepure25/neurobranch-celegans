"""Variance components and reliability ceilings for perturbation atlases.

The Phase 6/7 pipeline estimated a "split-half reliability ceiling" by repeatedly
splitting events, computing Spearman-Brown, and truncating negative values to
zero.  That estimator is unstable when a target has few events and it silently
biases the ceiling downward (Phase 7 reported normalised scores above 1).

This module replaces it with a variance-corrected estimator and, importantly,
distinguishes two ceilings that answer two different questions:

``repeat_reliability``
    How consistent is a target's response profile when the *same animal* is
    stimulated repeatedly?  This bounds claims about a single animal.

``single_animal_reliability``
    How much of one animal's profile variance is reproducible signal rather
    than sampling noise (shared target signal against sampling noise plus
    animal x responder interaction)?  This is the ceiling for claims about an
    individual animal and it is usually much lower.

``pooled_reliability``
    How reliable is a profile averaged over ``n_eval`` animals?  Phases 6-8
    score pair means pooled over the animals in a test fold, so the relevant
    ceiling is the pooled one, not ``single_animal_reliability``.  The module
    exposes the full curve so that every metric can be paired with the ceiling
    that matches its estimand.

Both are estimated on the observed scale.  Aggregate means carry a
target-level 95% interval; per-target estimates are reported with animal
counts instead of unstable single-target intervals.  The attenuation ceiling
is ``sqrt(reliability)``: the maximum correlation between the observed profile
and any error-free predictor.

Nothing here is a causal estimator.  A high ceiling means the measurement is
repeatable, not that the response is mechanistically understood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ANIMAL_COLUMN = "dataset_id"


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _safe_var(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan")
    return float(np.var(finite, ddof=1))


def _var_or_zero(values: np.ndarray) -> float:
    value = _safe_var(values)
    return 0.0 if not np.isfinite(value) else max(value, 0.0)


def _resolve_columns(
    frame: pd.DataFrame,
    *,
    response: str,
    animal: str,
) -> tuple[pd.DataFrame, str]:
    needed = {"target", "neuron", "event_id", response}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"reliability input missing columns {sorted(missing)}")
    work = frame.copy()
    if animal not in work.columns:
        if DEFAULT_ANIMAL_COLUMN in work.columns:
            animal = DEFAULT_ANIMAL_COLUMN
        else:
            work[animal] = "animal0"
    work = work.rename(columns={response: "_y"})
    work["_animal"] = work[animal].astype(str)
    work["_target"] = work["target"].astype(str)
    work["_neuron"] = work["neuron"].astype(str)
    return work, animal


# ---------------------------------------------------------------------------
# Cell summaries
# ---------------------------------------------------------------------------


def cell_summaries(
    frame: pd.DataFrame,
    *,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    one_event_variance: float | None = None,
    event_variance: float | None = None,
) -> pd.DataFrame:
    """Per (animal, target, responder) mean, count and within-cell variance.

    ``one_event_variance`` replaces the historical zero variance of cells with
    one event.  ``event_variance`` replaces every cell variance with a common
    per-event variance; both are audit/sensitivity options and leave the
    default estimator unchanged.  The returned ``var`` column is a variance of
    one event, so the sampling variance of a cell mean is ``var / n``.
    """
    if one_event_variance is not None and event_variance is not None:
        raise ValueError("set at most one of one_event_variance and event_variance")
    for value in (one_event_variance, event_variance):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("variance overrides must be finite and non-negative")
    work, _ = _resolve_columns(frame, response=response, animal=animal)
    cells = (
        work.groupby(["_animal", "_target", "_neuron"], dropna=False)["_y"]
        .agg(n="count", mean="mean", var="var")
        .reset_index()
    )
    cells["var"] = cells["var"].fillna(0.0)
    if one_event_variance is not None:
        cells.loc[cells["n"] <= 1, "var"] = float(one_event_variance)
    if event_variance is not None:
        cells["var"] = float(event_variance)
    return cells


# ---------------------------------------------------------------------------
# Repeat-stimulation reliability (within animal)
# ---------------------------------------------------------------------------


def _repeat_from_cells(
    cells: pd.DataFrame,
    *,
    min_events: int,
    min_responders: int,
) -> dict[str, float] | None:
    """Repeat reliability and counts for one target's cell table."""
    subset = cells[cells["n"] >= min_events]
    if subset.empty:
        return None
    true_sum = 0.0
    obs_sum = 0.0
    usable = 0
    for _animal_name, agroup in subset.groupby("_animal", sort=False):
        if len(agroup) < min_responders:
            continue
        means = agroup["mean"].to_numpy(dtype=float)
        sampling = agroup["var"].to_numpy(dtype=float) / np.maximum(agroup["n"].to_numpy(dtype=float), 1.0)
        observed_var = _safe_var(means)
        sampling_var = _nanmean(sampling)
        if not np.isfinite(observed_var) or observed_var <= 0 or not np.isfinite(sampling_var):
            continue
        true_sum += max(observed_var - sampling_var, 0.0)
        obs_sum += observed_var
        usable += 1
    if usable == 0 or obs_sum <= 0:
        return None
    reliability = float(np.clip(true_sum / obs_sum, 0.0, 1.0))
    return {
        "n_animals": float(usable),
        "n_responders": float(subset["_neuron"].nunique()),
        "n_events": float(subset["n"].sum()),
        "repeat_reliability": reliability,
        "repeat_ceiling": float(np.sqrt(reliability)),
    }


def repeat_reliability_by_target(
    frame: pd.DataFrame,
    *,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 2,
    min_responders: int = 3,
) -> pd.DataFrame:
    """Variance-corrected profile reliability within each animal, pooled per target.

    For one animal and target with responder means ``m_r`` (``r = 1..R``):

    ``V = Var_r(m_r)`` is the observed profile variance and
    ``S = mean_r(var_r / n_r)`` is the average sampling variance of those means.
    The error-free profile variance is ``V - S`` and the reliability is
    ``(V - S) / V``.  This is the finite-sample correction that split-half
    Spearman-Brown approximates noisily.
    """
    cells = cell_summaries(frame, response=response, animal=animal)
    rows: list[dict[str, Any]] = []
    for target, tgroup in cells.groupby("_target", sort=False):
        result = _repeat_from_cells(tgroup, min_events=min_events, min_responders=min_responders)
        if result is None:
            continue
        rows.append({"target": target, **result})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Across-animal generalisation reliability
# ---------------------------------------------------------------------------


def _nan_column_mean(values: np.ndarray) -> np.ndarray:
    """Column means that return NaN (not a warning) for all-missing columns."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    totals = np.where(finite, values, 0.0).sum(axis=0)
    out = np.full(values.shape[1], np.nan)
    np.divide(totals, counts, out=out, where=counts > 0)
    return out


def _nan_column_var(values: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Column variances, NaN where there are too few finite observations."""
    counts = np.isfinite(values).sum(axis=0)
    out = np.full(values.shape[1], np.nan)
    ok = counts > ddof
    if ok.any():
        out[ok] = np.nanvar(values[:, ok], axis=0, ddof=ddof)
    return out


@dataclass(frozen=True)
class ProfileComponents:
    """Variance components behind one target's reliability curve.

    ``signal`` is the error-free variance of the target profile across
    responders, ``noise`` is the mean sampling variance of one animal's
    responder means, and ``interaction`` is the animal x responder variance
    (idiosyncratic animal profiles).  The reliability of a profile averaged
    over ``n_eval`` animals is::

        signal / (signal + (noise + interaction) / n_eval)

    The formula is a moment approximation, not a REML estimate.  It is exact
    for the additive model it assumes and descriptive otherwise.
    """

    signal: float
    noise: float
    interaction: float
    n_animals: int
    n_responders: int

    def reliability_at(self, n_eval: int) -> float:
        if n_eval < 1:
            raise ValueError("n_eval must be >= 1")
        denominator = self.signal + (self.noise + self.interaction) / float(n_eval)
        if not np.isfinite(denominator) or denominator <= 0:
            return 0.0
        return float(np.clip(self.signal / denominator, 0.0, 1.0))

    @property
    def pooled_reliability(self) -> float:
        """Reliability of the profile averaged over the observed animals."""
        return self.reliability_at(self.n_animals)

    @property
    def single_animal_reliability(self) -> float:
        """Reliability of one animal's profile (``n_eval = 1``)."""
        return self.reliability_at(1)

    @property
    def ceiling(self) -> float:
        """Attenuation ceiling ``sqrt(reliability)`` at the observed animal count."""
        return float(np.sqrt(self.pooled_reliability))


def _components_from_arrays(
    values: np.ndarray,
    noise: np.ndarray,
    *,
    min_animals: int,
    min_responder_coverage: float = 0.0,
) -> ProfileComponents | None:
    """Components from an animal x responder matrix, allowing duplicate rows.

    This is the single source of truth for the moment estimator; the bootstrap
    calls it with resampled rows and columns.
    """
    if values.ndim != 2 or values.shape[1] < 3:
        return None
    if min_responder_coverage > 0:
        total_animals = values.shape[0]
        coverage = np.isfinite(values).sum(axis=0) / max(total_animals, 1)
        keep = coverage >= min_responder_coverage
        if keep.sum() < 3:
            return None
        values = values[:, keep]
        noise = noise[:, keep]
    finite_rows = np.isfinite(values).any(axis=1)
    values = values[finite_rows]
    noise = noise[finite_rows]
    if values.shape[0] < min_animals:
        return None

    pooled = _nan_column_mean(values)
    counts = np.isfinite(values).sum(axis=0)
    pooled_sampling = np.nansum(noise, axis=0) / np.maximum(counts, 1) ** 2
    observed_var = _safe_var(pooled)
    sampling_var = _nanmean(pooled_sampling)
    if not np.isfinite(observed_var) or observed_var <= 0 or not np.isfinite(sampling_var):
        return None
    total_r = _nan_column_var(values, ddof=1)
    noise_r = _nan_column_mean(noise)
    interaction_r = np.maximum(total_r - noise_r, 0.0)
    noise_mean = _nanmean(noise_r)
    interaction_mean = _nanmean(interaction_r)
    if not np.isfinite(noise_mean) or not np.isfinite(interaction_mean):
        return None
    n_animals = int(values.shape[0])
    # E[V] = signal + (noise + interaction)/A and S = noise/A, so V - S still
    # contains interaction/A.  Subtract it to avoid counting the animal x
    # responder variance twice.
    signal = max(observed_var - sampling_var - interaction_mean / n_animals, 0.0)
    return ProfileComponents(
        signal=float(signal),
        noise=float(noise_mean),
        interaction=float(interaction_mean),
        n_animals=n_animals,
        n_responders=int(values.shape[1]),
    )


def _profile_from_cells(
    cells: pd.DataFrame,
    *,
    min_events: int,
    min_animals: int,
    min_responder_coverage: float = 0.0,
) -> ProfileComponents | None:
    """Estimate the signal/noise/interaction components for one target.

    The pooled profile variance is ``V = Var_r(mean_a m_ar)`` and its sampling
    variance is ``S = mean_r(sum_a var_ar / n_ar) / A**2``.  The error-free
    profile variance is ``V - S``.  The interaction term is the observed
    between-animal variance minus the mean sampling variance, floored at zero.
    """
    subset = cells[cells["n"] >= min_events]
    if subset.empty:
        return None
    means = subset.pivot_table(index="_animal", columns="_neuron", values="mean", aggfunc="mean")
    sampling_cells = subset.assign(se2=subset["var"] / np.maximum(subset["n"], 1.0))
    sampling = sampling_cells.pivot_table(index="_animal", columns="_neuron", values="se2", aggfunc="mean")
    means = means.dropna(axis=1, how="all")
    sampling = sampling.reindex(columns=means.columns)
    if means.shape[0] < min_animals or means.shape[1] < 3:
        return None
    return _components_from_arrays(
        means.to_numpy(dtype=float),
        sampling.to_numpy(dtype=float),
        min_animals=min_animals,
        min_responder_coverage=min_responder_coverage,
    )


def animal_reliability_by_target(
    frame: pd.DataFrame,
    *,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
    seed: int = 17,
    min_responder_coverage: float = 0.0,
    one_event_variance: float | None = None,
    event_variance: float | None = None,
) -> pd.DataFrame:
    """Across-animal reliability of each target's response profile.

    The across-animal facet does not require repeated stimulation: one
    measurement per animal is a valid observation of that animal's profile.
    ``min_events`` therefore defaults to 1 here, unlike the within-animal
    repeat estimator.  ``seed`` is retained for API stability; the estimator
    is deterministic.
    """
    del seed  # deterministic estimator; parameter kept for compatibility
    components = profile_components_by_target(
        frame,
        response=response,
        animal=animal,
        min_events=min_events,
        min_animals=min_animals,
        min_responder_coverage=min_responder_coverage,
        one_event_variance=one_event_variance,
        event_variance=event_variance,
    )
    if components.empty:
        return components
    pooled = components["pooled_reliability"].to_numpy(dtype=float)
    single = components["single_animal_reliability"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "target": components["target"].to_numpy(),
            "n_animals": components["n_animals"].to_numpy(),
            "n_responders": components["n_responders"].to_numpy(),
            "animal_mean_reliability": pooled,
            "animal_mean_ceiling": np.sqrt(pooled),
            "single_animal_reliability": single,
            "animal_ceiling": np.sqrt(single),
        }
    )


def profile_components_by_target(
    frame: pd.DataFrame,
    *,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
    min_responder_coverage: float = 0.0,
    one_event_variance: float | None = None,
    event_variance: float | None = None,
) -> pd.DataFrame:
    """Signal, noise and interaction components for every eligible target.

    Returns one row per target with ``n_animals``, ``n_responders``, ``signal``,
    ``noise``, ``interaction``, ``pooled_reliability`` (at the observed animal
    count) and ``single_animal_reliability``.  Targets with fewer than
    ``min_animals`` animals or fewer than three responders are omitted.

    ``one_event_variance`` and ``event_variance`` are passed to
    :func:`cell_summaries`; the default is the historical zero-variance
    treatment for one-event cells.
    """
    cells = cell_summaries(
        frame,
        response=response,
        animal=animal,
        one_event_variance=one_event_variance,
        event_variance=event_variance,
    )
    return profile_components_from_cells(
        cells,
        min_events=min_events,
        min_animals=min_animals,
        min_responder_coverage=min_responder_coverage,
    )


def profile_components_from_cells(
    cells: pd.DataFrame,
    *,
    min_events: int = 1,
    min_animals: int = 6,
    min_responder_coverage: float = 0.0,
) -> pd.DataFrame:
    """Signal/noise components from a precomputed cell-summary table.

    ``cells`` must contain the ``_animal``, ``_target``, ``_neuron``, ``n``,
    ``mean`` and ``var`` columns produced by :func:`cell_summaries`.  The
    function exists so bootstrap and audit code can resample animals without
    re-reading the event table.
    """
    required = {"_animal", "_target", "_neuron", "n", "mean", "var"}
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"cell table missing columns {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for target, tgroup in cells.groupby("_target", sort=False):
        components = _profile_from_cells(
            tgroup,
            min_events=min_events,
            min_animals=min_animals,
            min_responder_coverage=min_responder_coverage,
        )
        if components is None:
            continue
        rows.append(
            {
                "target": target,
                "n_animals": components.n_animals,
                "n_responders": components.n_responders,
                "signal": components.signal,
                "noise": components.noise,
                "interaction": components.interaction,
                "pooled_reliability": components.pooled_reliability,
                "single_animal_reliability": components.single_animal_reliability,
            }
        )
    return pd.DataFrame(rows)


def pooled_reliability_by_target(
    frame: pd.DataFrame,
    *,
    n_eval: int | None = None,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
    min_responder_coverage: float = 0.0,
) -> pd.DataFrame:
    """Reliability of a profile averaged over ``n_eval`` animals.

    ``n_eval=None`` uses each target's observed animal count.  A requested
    ``n_eval`` larger than the observed count is flagged ``extrapolated`` and
    must not be used in a confirmatory claim.
    """
    components = profile_components_by_target(
        frame,
        response=response,
        animal=animal,
        min_events=min_events,
        min_animals=min_animals,
    )
    columns = ["target", "n_eval", "reliability", "ceiling", "extrapolated", "n_animals"]
    if components.empty:
        return pd.DataFrame(columns=columns)
    requested = components["n_animals"].to_numpy(dtype=int) if n_eval is None else np.full(len(components), int(n_eval))
    if np.any(requested < 1):
        raise ValueError("n_eval must be >= 1")
    signal = components["signal"].to_numpy(dtype=float)
    error = (components["noise"] + components["interaction"]).to_numpy(dtype=float)
    reliability = signal / (signal + error / requested)
    reliability = np.clip(reliability, 0.0, 1.0)
    return pd.DataFrame(
        {
            "target": components["target"].to_numpy(),
            "n_eval": requested,
            "reliability": reliability,
            "ceiling": np.sqrt(reliability),
            "extrapolated": requested > components["n_animals"].to_numpy(dtype=int),
            "n_animals": components["n_animals"].to_numpy(),
        }
    )


def reliability_curve(
    frame: pd.DataFrame,
    *,
    n_grid: tuple[int, ...] = (1, 2, 5, 10, 22, 55, 110),
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
) -> pd.DataFrame:
    """Long-form reliability curve: one row per target and ``n_eval``."""
    components = profile_components_by_target(
        frame,
        response=response,
        animal=animal,
        min_events=min_events,
        min_animals=min_animals,
    )
    columns = ["target", "n_eval", "reliability", "ceiling", "extrapolated", "n_animals"]
    if components.empty:
        return pd.DataFrame(columns=columns)
    signal = components["signal"].to_numpy(dtype=float)
    error = (components["noise"] + components["interaction"]).to_numpy(dtype=float)
    observed = components["n_animals"].to_numpy(dtype=int)
    frames = []
    for n_eval in n_grid:
        if int(n_eval) < 1:
            raise ValueError("n_eval must be >= 1")
        reliability = np.clip(signal / (signal + error / float(n_eval)), 0.0, 1.0)
        frames.append(
            pd.DataFrame(
                {
                    "target": components["target"].to_numpy(),
                    "n_eval": np.full(len(components), int(n_eval)),
                    "reliability": reliability,
                    "ceiling": np.sqrt(reliability),
                    "extrapolated": int(n_eval) > observed,
                    "n_animals": observed,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def summarise_components(components: pd.DataFrame, *, fold_size: int = 22) -> dict[str, float]:
    """Aggregate the per-target components at a chosen evaluation-animal count.

    ``fold_size`` is the number of animals whose responses are pooled into a
    pair mean in a Phase 6-8 test fold.  The function reports both the mean of
    per-target ceilings and the ceiling of the mean reliability because they are
    different quantities and must not be used interchangeably.
    """
    if components.empty:
        return {
            "n_pooled_targets": 0.0,
            "fold_size": float(fold_size),
            "pooled_reliability_at_fold": float("nan"),
            "pooled_ceiling_at_fold": float("nan"),
            "pooled_reliability_observed": float("nan"),
            "mean_target_ceiling": float("nan"),
            "ceiling_of_mean_reliability": float("nan"),
        }
    signal = components["signal"].to_numpy(dtype=float)
    error = (components["noise"] + components["interaction"]).to_numpy(dtype=float)
    at_fold = np.clip(signal / (signal + error / float(fold_size)), 0.0, 1.0)
    at_observed = np.clip(
        signal / (signal + error / components["n_animals"].to_numpy(dtype=float)),
        0.0,
        1.0,
    )
    mean_fold, _, _ = _mean_ci(at_fold)
    return {
        "n_pooled_targets": float(np.isfinite(at_fold).sum()),
        "fold_size": float(fold_size),
        "pooled_reliability_at_fold": float(mean_fold),
        "pooled_ceiling_at_fold": float(np.sqrt(mean_fold)) if np.isfinite(mean_fold) else float("nan"),
        "pooled_reliability_observed": float(np.nanmean(at_observed)),
        "mean_target_ceiling": float(np.nanmean(np.sqrt(at_fold))),
        "ceiling_of_mean_reliability": float(np.sqrt(np.nanmean(at_fold))),
    }


def _reliability_at_n(signal: np.ndarray, error: np.ndarray, n_eval: np.ndarray | float) -> np.ndarray:
    """Reliability for a fixed evaluation-animal count, clipped to [0, 1]."""
    denominator = signal + error / np.asarray(n_eval, dtype=float)
    values = np.zeros_like(signal, dtype=float)
    np.divide(signal, denominator, out=values, where=denominator > 0)
    return np.clip(values, 0.0, 1.0)


def estimand_table(components: pd.DataFrame, *, fold_size: int = 22) -> pd.DataFrame:
    """Per-target values for the three Phase 15 estimands.

    ``R_obs`` uses each target's observed animal count; ``R_22`` fixes the count
    at ``fold_size`` and is therefore a counterfactual for targets with fewer
    animals; ``R_id`` is the observed-count value only for the identified subset
    with at least ``fold_size`` animals.  The boolean columns make the
    extrapolation and population membership explicit.
    """
    columns = [
        "target",
        "n_animals",
        "n_responders",
        "signal",
        "noise",
        "interaction",
        "reliability_obs",
        "ceiling_obs",
        "reliability_22",
        "ceiling_22",
        "reliability_id",
        "ceiling_id",
        "in_identified_subset",
        "extrapolated_22",
    ]
    if components.empty:
        return pd.DataFrame(columns=columns)
    if fold_size < 1:
        raise ValueError("fold_size must be >= 1")
    signal = components["signal"].to_numpy(dtype=float)
    error = (components["noise"] + components["interaction"]).to_numpy(dtype=float)
    n_animals = components["n_animals"].to_numpy(dtype=int)
    observed = _reliability_at_n(signal, error, n_animals)
    at_fold = _reliability_at_n(signal, error, float(fold_size))
    in_subset = n_animals >= int(fold_size)
    identified = np.where(in_subset, observed, np.nan)
    return pd.DataFrame(
        {
            "target": components["target"].to_numpy(),
            "n_animals": n_animals,
            "n_responders": components["n_responders"].to_numpy(dtype=int),
            "signal": signal,
            "noise": components["noise"].to_numpy(dtype=float),
            "interaction": components["interaction"].to_numpy(dtype=float),
            "reliability_obs": observed,
            "ceiling_obs": np.sqrt(observed),
            "reliability_22": at_fold,
            "ceiling_22": np.sqrt(at_fold),
            "reliability_id": identified,
            "ceiling_id": np.sqrt(identified),
            "in_identified_subset": in_subset,
            "extrapolated_22": ~in_subset,
        }
    )


def _estimand_block(values: np.ndarray, *, n_extrapolated: int = 0) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "estimate": float("nan"),
            "ceiling": float("nan"),
            "n_targets": 0.0,
            "n_extrapolated": float(n_extrapolated),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }
    mean = float(finite.mean())
    return {
        "estimate": mean,
        "ceiling": float(np.sqrt(max(mean, 0.0))),
        "n_targets": float(finite.size),
        "n_extrapolated": float(n_extrapolated),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def summarise_reliability_estimands(components: pd.DataFrame, *, fold_size: int = 22) -> dict[str, Any]:
    """Aggregate the three pre-specified Phase 15 estimands.

    The returned dictionary always contains all three blocks.  ``R_22`` records
    how many targets are extrapolated so that a consumer cannot mistake it for
    an observed-data quantity.  ``R_id`` has a different target population and
    is not a replacement for ``R_obs``.
    """
    table = estimand_table(components, fold_size=fold_size)
    if table.empty:
        return {
            "fold_size": float(fold_size),
            "R_obs": _estimand_block(np.array([], dtype=float)),
            "R_22": _estimand_block(np.array([], dtype=float)),
            "R_id": _estimand_block(np.array([], dtype=float)),
            "n_animals_total": 0.0,
        }
    return {
        "fold_size": float(fold_size),
        "R_obs": _estimand_block(table["reliability_obs"].to_numpy(dtype=float)),
        "R_22": _estimand_block(
            table["reliability_22"].to_numpy(dtype=float),
            n_extrapolated=int(table["extrapolated_22"].sum()),
        ),
        "R_id": _estimand_block(table["reliability_id"].to_numpy(dtype=float)),
        "n_animals_total": float(table["n_animals"].sum()),
    }


def summarise_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a reliability curve by ``n_eval``.

    The interval is a target-level normal approximation and therefore
    descriptive: targets share animals.  Use the animal jackknife in
    ``benchmark.robustness`` for a cluster-robust interval.
    """
    columns = ["n_eval", "n_targets", "n_extrapolated", "mean_reliability", "ci_low", "ci_high", "median_reliability", "ceiling"]
    if curve.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, float]] = []
    for n_eval, group in curve.groupby("n_eval", sort=True):
        values = group["reliability"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        mean = float(finite.mean()) if finite.size else float("nan")
        if finite.size >= 3:
            _, low, high = _mean_ci(finite)
        else:
            low = high = float("nan")
        rows.append(
            {
                "n_eval": float(n_eval),
                "n_targets": float(finite.size),
                "n_extrapolated": float(group["extrapolated"].sum()),
                "mean_reliability": mean,
                "ci_low": low,
                "ci_high": high,
                "median_reliability": float(np.nanmedian(values)),
                "ceiling": float(np.sqrt(mean)) if np.isfinite(mean) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary with target-level uncertainty
# ---------------------------------------------------------------------------


def _mean_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and normal-approximation CI across targets (target-level SE).

    Targets share animals, so this interval describes the target population
    rather than a cluster-robust population interval.  It is used only for the
    aggregate ceiling claim and is labelled accordingly in the report.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return float("nan"), float("nan"), float("nan")
    mean = float(finite.mean())
    se = float(finite.std(ddof=1) / np.sqrt(finite.size))
    z = 1.959963984540054
    return mean, mean - z * se, mean + z * se


# ---------------------------------------------------------------------------
# Variance decomposition (descriptive moments, not REML)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarianceComponents:
    """Additive two-way moment decomposition of pair-level responses.

    ``animal`` and ``target`` are additive main effects; ``pair`` is the
    target x responder residual after removing them, corrected for sampling
    error; ``within_event`` is the pooled within-cell variance.  This is a
    descriptive decomposition for design calculations, not a likelihood fit.
    """

    within_event: float
    pair: float
    target: float
    animal: float
    total_pair_mean: float
    n_animals: int
    n_targets: int
    n_pairs: int
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "within_event": self.within_event,
            "pair": self.pair,
            "target": self.target,
            "animal": self.animal,
            "total_pair_mean": self.total_pair_mean,
            "n_animals": self.n_animals,
            "n_targets": self.n_targets,
            "n_pairs": self.n_pairs,
            "notes": self.notes,
        }


def decompose_variance(
    frame: pd.DataFrame,
    *,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 2,
) -> VarianceComponents:
    """Moment decomposition of between-animal, between-target and pair variance."""
    cells = cell_summaries(frame, response=response, animal=animal)
    cells = cells[cells["n"] >= min_events]
    if cells.empty:
        return VarianceComponents(*(float("nan"),) * 5, 0, 0, 0)
    means = cells["mean"].to_numpy(dtype=float)
    grand = float(np.nanmean(means))
    animal_means = cells.groupby("_animal")["mean"].mean()
    target_means = cells.groupby("_target")["mean"].mean()
    a_effect = animal_means - grand
    t_effect = target_means - grand
    lookup_a = cells["_animal"].map(a_effect).to_numpy(dtype=float)
    lookup_t = cells["_target"].map(t_effect).to_numpy(dtype=float)
    residual = means - grand - lookup_a - lookup_t
    sampling = float(np.nanmean(cells["var"].to_numpy(dtype=float) / np.maximum(cells["n"].to_numpy(dtype=float), 1)))
    pair_var = max(_safe_var(residual) - sampling, 0.0) if np.isfinite(_safe_var(residual)) else 0.0
    within = float(np.nanmean(cells["var"].to_numpy(dtype=float)))
    return VarianceComponents(
        within_event=within,
        pair=float(pair_var),
        target=_var_or_zero(t_effect.to_numpy(dtype=float)),
        animal=_var_or_zero(a_effect.to_numpy(dtype=float)),
        total_pair_mean=_var_or_zero(means),
        n_animals=int(cells["_animal"].nunique()),
        n_targets=int(cells["_target"].nunique()),
        n_pairs=int(len(cells)),
        notes={"estimator": "additive two-way moments", "sampling_variance": sampling},
    )


# ---------------------------------------------------------------------------
# Reliability table facade
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilitySummary:
    """Aggregate ceiling summary used by the benchmark card and design engine."""

    n_targets: int
    mean_repeat_reliability: float
    median_repeat_reliability: float
    mean_animal_reliability: float
    median_animal_reliability: float
    mean_repeat_ceiling: float
    mean_animal_ceiling: float
    fraction_repeat_above_half: float
    fraction_animal_above_half: float
    repeat_ci_low: float = float("nan")
    repeat_ci_high: float = float("nan")
    animal_ci_low: float = float("nan")
    animal_ci_high: float = float("nan")
    p90_animal_reliability: float = float("nan")
    n_repeat_targets: int = 0
    n_animal_targets: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "n_targets": float(self.n_targets),
            "mean_repeat_reliability": self.mean_repeat_reliability,
            "median_repeat_reliability": self.median_repeat_reliability,
            "mean_animal_reliability": self.mean_animal_reliability,
            "median_animal_reliability": self.median_animal_reliability,
            "mean_repeat_ceiling": self.mean_repeat_ceiling,
            "mean_animal_ceiling": self.mean_animal_ceiling,
            "fraction_repeat_above_half": self.fraction_repeat_above_half,
            "fraction_animal_above_half": self.fraction_animal_above_half,
            "repeat_ci_low": self.repeat_ci_low,
            "repeat_ci_high": self.repeat_ci_high,
            "animal_ci_low": self.animal_ci_low,
            "animal_ci_high": self.animal_ci_high,
            "p90_animal_reliability": self.p90_animal_reliability,
            "n_repeat_targets": float(self.n_repeat_targets),
            "n_animal_targets": float(self.n_animal_targets),
        }


def summarise_reliability(table: pd.DataFrame) -> ReliabilitySummary:
    """Summarise a reliability table, ignoring missing ceilings."""
    if table.empty:
        return ReliabilitySummary(0, *(float("nan"),) * 8)
    rep = table["repeat_reliability"].to_numpy(dtype=float) if "repeat_reliability" in table else np.array([])
    ani = (
        table["single_animal_reliability"].to_numpy(dtype=float)
        if "single_animal_reliability" in table
        else np.array([])
    )
    rep_ok = rep[np.isfinite(rep)]
    ani_ok = ani[np.isfinite(ani)]
    rep_mean, rep_low, rep_high = _mean_ci(rep_ok)
    ani_mean, ani_low, ani_high = _mean_ci(ani_ok)
    return ReliabilitySummary(
        n_targets=int(len(table)),
        mean_repeat_reliability=rep_mean,
        median_repeat_reliability=float(np.median(rep_ok)) if rep_ok.size else float("nan"),
        mean_animal_reliability=ani_mean,
        median_animal_reliability=float(np.median(ani_ok)) if ani_ok.size else float("nan"),
        mean_repeat_ceiling=float(np.sqrt(max(rep_mean, 0.0))) if np.isfinite(rep_mean) else float("nan"),
        mean_animal_ceiling=float(np.sqrt(max(ani_mean, 0.0))) if np.isfinite(ani_mean) else float("nan"),
        fraction_repeat_above_half=float(np.mean(rep_ok >= 0.5)) if rep_ok.size else float("nan"),
        fraction_animal_above_half=float(np.mean(ani_ok >= 0.5)) if ani_ok.size else float("nan"),
        repeat_ci_low=rep_low,
        repeat_ci_high=rep_high,
        animal_ci_low=ani_low,
        animal_ci_high=ani_high,
        p90_animal_reliability=float(np.quantile(ani_ok, 0.9)) if ani_ok.size else float("nan"),
        n_repeat_targets=int(rep_ok.size),
        n_animal_targets=int(ani_ok.size),
    )


__all__ = [
    "ReliabilitySummary",
    "VarianceComponents",
    "animal_reliability_by_target",
    "cell_summaries",
    "decompose_variance",
    "estimand_table",
    "profile_components_by_target",
    "profile_components_from_cells",
    "repeat_reliability_by_target",
    "summarise_components",
    "summarise_reliability",
    "summarise_reliability_estimands",
]
