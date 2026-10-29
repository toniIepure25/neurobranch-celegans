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
    animal x responder interaction)?  This is the ceiling that matters for
    animal-held-out claims (Phases 6-8) and it is usually much lower.

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
) -> pd.DataFrame:
    """Per (animal, target, responder) mean, count and within-cell variance."""
    work, _ = _resolve_columns(frame, response=response, animal=animal)
    cells = (
        work.groupby(["_animal", "_target", "_neuron"], dropna=False)["_y"]
        .agg(n="count", mean="mean", var="var")
        .reset_index()
    )
    cells["var"] = cells["var"].fillna(0.0)
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


def _animal_from_cells(
    cells: pd.DataFrame,
    *,
    min_events: int,
    min_animals: int,
) -> dict[str, float] | None:
    """Variance-corrected across-animal reliability for one target.

    Two quantities are returned because they answer different questions:

    ``animal_mean_reliability``
        How reliable is the *pooled* profile (mean over animals) as a measure
        of the latent target profile?  This is the ceiling for the pooled
        pair-level metrics used in Phases 6-7.

    ``single_animal_reliability``
        How much of one animal's profile variance is reproducible signal
        rather than sampling noise?  This is the ceiling for animal-held-out
        event-level metrics in Phase 8, and it is almost always lower.

    Split-half correlation is deliberately not used here: resampling animals
    that appear in both halves inflates the split-half correlation.  The
    variance correction below is stable and has no such artefact.
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

    values = means.to_numpy(dtype=float)
    noise = sampling.to_numpy(dtype=float)
    n_animals = values.shape[0]

    # Pooled profile: variance of the animal-mean profile and its sampling error.
    pooled = _nan_column_mean(values)
    pooled_sampling = np.nansum(noise, axis=0) / (n_animals**2)
    observed_var = _safe_var(pooled)
    sampling_var = _nanmean(pooled_sampling)
    if not np.isfinite(observed_var) or observed_var <= 0 or not np.isfinite(sampling_var):
        return None
    mean_reliability = float(np.clip((observed_var - sampling_var) / observed_var, 0.0, 1.0))

    # Single animal: shared signal relative to sampling noise plus
    # animal x responder interaction (idiosyncratic animal profiles).
    total_r = _nan_column_var(values, ddof=1)
    noise_r = np.nanmean(noise, axis=0)
    interaction_r = np.maximum(total_r - noise_r, 0.0)
    signal = max(observed_var - sampling_var, 0.0)
    denominator = signal + _nanmean(noise_r) + _nanmean(interaction_r)
    if not np.isfinite(denominator) or denominator <= 0:
        single = 0.0
    else:
        single = float(np.clip(signal / denominator, 0.0, 1.0))

    return {
        "n_animals": float(n_animals),
        "n_responders": float(means.shape[1]),
        "animal_mean_reliability": mean_reliability,
        "animal_mean_ceiling": float(np.sqrt(mean_reliability)),
        "single_animal_reliability": single,
        "animal_ceiling": float(np.sqrt(single)),
    }


def animal_reliability_by_target(
    frame: pd.DataFrame,
    *,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
    seed: int = 17,
) -> pd.DataFrame:
    """Across-animal reliability of each target's response profile.

    The across-animal facet does not require repeated stimulation: one
    measurement per animal is a valid observation of that animal's profile.
    ``min_events`` therefore defaults to 1 here, unlike the within-animal
    repeat estimator.  ``seed`` is retained for API stability; the estimator
    is deterministic.
    """
    del seed  # deterministic estimator; parameter kept for compatibility
    cells = cell_summaries(frame, response=response, animal=animal)
    rows: list[dict[str, Any]] = []
    for target, tgroup in cells.groupby("_target", sort=False):
        result = _animal_from_cells(tgroup, min_events=min_events, min_animals=min_animals)
        if result is None:
            continue
        rows.append({"target": target, **result})
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
        return VarianceComponents(*(float("nan"),) * 5, 0, 0, 0)  # type: ignore[arg-type]
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
        return ReliabilitySummary(0, *(float("nan"),) * 8)  # type: ignore[arg-type]
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
    "repeat_reliability_by_target",
    "summarise_reliability",
]




