"""Robustness checks for the metric-aligned reliability ceilings.

Three questions are answered here, each with a different resampling unit:

``summary_sensitivity``
    Does the ceiling depend on how a trial response is summarised?

``animal_half_replication``
    Are per-target reliability estimates stable when the animal sample is
    split in half?  This tests the estimator, not the biology.

``animal_jackknife``
    What is the cluster-robust uncertainty of the aggregate pooled ceiling
    when animals, not targets, are the resampling unit?

None of these functions makes a causal claim.  They quantify measurement
stability and estimator uncertainty only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neurobranch.benchmark.variance import (
    DEFAULT_ANIMAL_COLUMN,
    _nanmean,
    _profile_from_cells,
    _safe_var,
    cell_summaries,
    profile_components_by_target,
)

SUMMARY_COLUMNS = ("response_mean", "response_integral", "response_residual", "response_peak_time")


def _leave_one_out_components(values: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-animal-out components for a sparse animal x responder matrix.

    Missing cells are expected: an animal need not respond in every neuron.
    Per-responder counts therefore replace the constant animal count used by
    the dense formula.
    """
    n_animals = values.shape[0]
    if n_animals < 3:
        raise ValueError("leave-one-out components need at least 3 animals")

    present = np.isfinite(values)
    counts = present.sum(axis=0)
    totals = np.nansum(values, axis=0)
    totals_sq = np.nansum(values**2, axis=0)

    noise_present = np.isfinite(noise) & present
    noise_totals = np.nansum(noise, axis=0)

    signal = np.full(n_animals, np.nan)
    noise_mean = np.full(n_animals, np.nan)
    interaction_mean = np.full(n_animals, np.nan)
    for j in range(n_animals):
        has_j = present[j]
        remaining = counts - has_j.astype(int)
        valid_mean = remaining >= 1
        valid_var = remaining >= 2
        if not np.any(valid_mean):
            continue
        sum_r = totals - np.where(has_j, values[j], 0.0)
        sumsq_r = totals_sq - np.where(has_j, values[j] ** 2, 0.0)
        mean_r = np.full(values.shape[1], np.nan)
        np.divide(sum_r, remaining, out=mean_r, where=valid_mean)
        observed = _safe_var(mean_r)
        if not np.isfinite(observed) or observed <= 0:
            continue

        noise_sum = noise_totals - np.where(noise_present[j], noise[j], 0.0)
        sampling_r = np.full(values.shape[1], np.nan)
        np.divide(noise_sum, remaining, out=sampling_r, where=valid_mean)
        pooled_sampling = np.full(values.shape[1], np.nan)
        np.divide(noise_sum, remaining**2, out=pooled_sampling, where=valid_mean)
        sampling_mean = _nanmean(pooled_sampling)
        if not np.isfinite(sampling_mean):
            continue

        variance_r = np.full(values.shape[1], np.nan)
        np.divide(
            sumsq_r - sum_r**2 / np.where(valid_var, remaining, np.nan),
            np.where(valid_var, remaining - 1, np.nan),
            out=variance_r,
            where=valid_var,
        )
        interaction_r = np.maximum(variance_r - sampling_r, 0.0)
        interaction_correction = _nanmean(
            interaction_r / np.where(valid_var, remaining, np.nan)
        )
        if not np.isfinite(interaction_correction):
            interaction_correction = 0.0
        signal_j = max(observed - sampling_mean - interaction_correction, 0.0)
        signal[j] = signal_j
        noise_mean[j] = _nanmean(sampling_r)
        interaction_mean[j] = _nanmean(interaction_r)
    return signal, noise_mean, interaction_mean


def animal_jackknife(
    frame: pd.DataFrame,
    *,
    fold_size: int = 22,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
    level: float = 0.95,
) -> dict[str, float]:
    """Delete-one-animal jackknife for the mean pooled reliability.

    The estimand is the mean over targets of the reliability of a profile
    pooled over ``fold_size`` animals.  Targets that do not contain the deleted
    animal contribute their full-sample value, which is the correct
    delete-one-cluster estimand.
    """
    if fold_size < 1:
        raise ValueError("fold_size must be >= 1")
    components = profile_components_by_target(
        frame,
        response=response,
        animal=animal,
        min_events=min_events,
        min_animals=min_animals,
    )
    cells = cell_summaries(frame, response=response, animal=animal)

    targets = list(components["target"]) if not components.empty else []
    animals = sorted(cells["_animal"].dropna().unique())
    animal_index = {name: index for index, name in enumerate(animals)}
    n_targets = len(targets)
    n_animals = len(animals)
    if n_targets == 0 or n_animals < 3:
        return {
            "estimate": float("nan"),
            "jackknife_mean": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_targets": float(n_targets),
            "n_animals": float(n_animals),
        }

    reliability = np.full((n_targets, n_animals), np.nan)
    target_position = {target: index for index, target in enumerate(targets)}
    for target in targets:
        group = components[components["target"] == target]
        if group.empty:
            continue
        row = group.iloc[0]
        full = float(
            np.clip(
                row["signal"] / (row["signal"] + (row["noise"] + row["interaction"]) / float(fold_size)),
                0.0,
                1.0,
            )
        )
        reliability[target_position[target], :] = full
    full_estimate = float(np.nanmean(reliability))

    for target, tgroup in cells.groupby("_target", sort=False):
        if target not in target_position:
            continue
        subset = tgroup[tgroup["n"] >= min_events]
        if subset.empty:
            continue
        means = subset.pivot_table(index="_animal", columns="_neuron", values="mean", aggfunc="mean")
        sampling_cells = subset.assign(se2=subset["var"] / np.maximum(subset["n"], 1.0))
        noise = sampling_cells.pivot_table(index="_animal", columns="_neuron", values="se2", aggfunc="mean")
        means = means.dropna(axis=1, how="all")
        noise = noise.reindex(index=means.index, columns=means.columns)
        if means.shape[0] < max(min_animals, 3) or means.shape[1] < 3:
            continue
        values = means.to_numpy(dtype=float)
        noise_values = noise.to_numpy(dtype=float)
        signal, noise_mean, interaction_mean = _leave_one_out_components(values, noise_values)
        for local_index, animal_name in enumerate(means.index):
            if animal_name not in animal_index:
                continue
            denominator = signal[local_index] + (noise_mean[local_index] + interaction_mean[local_index]) / float(fold_size)
            if np.isfinite(denominator) and denominator > 0:
                reliability[target_position[target], animal_index[animal_name]] = float(
                    np.clip(signal[local_index] / denominator, 0.0, 1.0)
                )

    column_means = np.nanmean(reliability, axis=0)
    finite = column_means[np.isfinite(column_means)]
    if finite.size < 3:
        return {
            "estimate": float("nan"),
            "jackknife_mean": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_targets": float(n_targets),
            "n_animals": float(finite.size),
        }
    estimate = full_estimate
    jackknife_mean = float(finite.mean())
    se = float(np.sqrt((finite.size - 1) / finite.size * np.sum((finite - jackknife_mean) ** 2)))
    z = float(stats.norm.ppf(0.5 + level / 2.0))

    ci_low = float(max(0.0, estimate - z * se))
    ci_high = float(min(1.0, estimate + z * se))
    return {
        "estimate": estimate,
        "jackknife_mean": jackknife_mean,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_targets": float(n_targets),
        "n_animals": float(finite.size),
    }


def summary_sensitivity(
    frame: pd.DataFrame,
    *,
    summaries: tuple[str, ...] = ("response_mean", "response_integral", "response_residual"),
    reference: str = "response_mean",
    fold_size: int = 22,
    min_animals: int = 6,
) -> pd.DataFrame:
    """Compare the fold-level pooled ceiling across response summaries.

    Each summary is rescaled only through the variance components, so the
    comparison is scale-free.  The rank correlation is computed on the
    per-target reliability values and therefore asks whether the *ordering* of
    targets is summary-dependent.
    """
    rows: list[dict[str, Any]] = []
    reference_values: pd.Series | None = None
    for summary in summaries:
        if summary not in frame.columns:
            continue
        components = profile_components_by_target(frame, response=summary, min_animals=min_animals)
        if components.empty:
            continue
        signal = components["signal"].to_numpy(dtype=float)
        error = (components["noise"] + components["interaction"]).to_numpy(dtype=float)
        reliability = np.clip(signal / (signal + error / float(fold_size)), 0.0, 1.0)
        series = pd.Series(reliability, index=components["target"].to_numpy(dtype=str)).sort_index()
        if summary == reference:
            reference_values = series
        rows.append(
            {
                "summary": summary,
                "n_targets": float(len(series)),
                "mean_reliability_at_fold": float(np.nanmean(reliability)),
                "ceiling_at_fold": float(np.sqrt(np.nanmean(reliability))),
                "median_reliability_at_fold": float(np.nanmedian(reliability)),
            }
        )
    table = pd.DataFrame(rows)
    if reference_values is not None and not table.empty:
        correlations = []
        for summary in table["summary"]:
            components = profile_components_by_target(frame, response=summary, min_animals=min_animals)
            signal = components["signal"].to_numpy(dtype=float)
            error = (components["noise"] + components["interaction"]).to_numpy(dtype=float)
            reliability = np.clip(signal / (signal + error / float(fold_size)), 0.0, 1.0)
            series = pd.Series(reliability, index=components["target"].to_numpy(dtype=str))
            joined = pd.concat([reference_values, series], axis=1, join="inner").dropna()
            correlations.append(
                float(stats.spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).statistic)
                if len(joined) >= 3
                else float("nan")
            )
        table["spearman_vs_reference"] = correlations
    else:
        table["spearman_vs_reference"] = np.nan
    return table


def animal_half_replication(
    frame: pd.DataFrame,
    *,
    n_splits: int = 25,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
    seed: int = 17,
    shrink: bool = False,
    min_responder_coverage: float = 0.0,
) -> dict[str, float]:
    """Split animals in half and correlate per-target single-animal reliability.

    Both halves are scored with the ``n_eval = 1`` formula, so the comparison
    is not confounded by the different number of animals per half.  Targets
    must be eligible in both halves.  With ``shrink=True`` each half is
    empirical-Bayes shrunk before correlation, which tests whether shrinkage
    stabilises the ranking rather than merely improving the point estimates.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    cells = cell_summaries(frame, response=response, animal=animal)
    animals = np.array(sorted(cells["_animal"].dropna().unique()))
    if animals.size < 2 * min_animals:
        return {
            "median_spearman": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_splits": 0.0,
            "median_targets": float("nan"),
        }
    groups = {target: group for target, group in cells.groupby("_target", sort=False)}
    rng = np.random.default_rng(seed)
    correlations: list[float] = []
    target_counts: list[int] = []
    for _ in range(n_splits):
        shuffled = rng.permutation(animals)
        halves = (set(shuffled[: shuffled.size // 2]), set(shuffled[shuffled.size // 2 :]))
        per_half: list[pd.DataFrame] = []
        for half in halves:
            rows: list[dict[str, float | str]] = []
            for target, group in groups.items():
                subset = group[group["_animal"].isin(half)]
                components = _profile_from_cells(
                    subset,
                    min_events=min_events,
                    min_animals=min_animals,
                    min_responder_coverage=min_responder_coverage,
                )
                if components is not None:
                    rows.append(
                        {
                            "target": target,
                            "single_animal_reliability": components.single_animal_reliability,
                            "n_animals": float(components.n_animals),
                        }
                    )
            per_half.append(pd.DataFrame(rows))
        first, second = per_half
        if first.empty or second.empty:
            continue
        if shrink:
            from neurobranch.benchmark.reliability import shrink_reliability

            first = shrink_reliability(first)
            second = shrink_reliability(second)
            column = "single_animal_reliability_shrunk"
        else:
            column = "single_animal_reliability"
        joined = first[["target", column]].merge(
            second[["target", column]], on="target", suffixes=("_first", "_second")
        ).dropna()
        if len(joined) >= 3:
            correlations.append(
                float(
                    stats.spearmanr(
                        joined[f"{column}_first"], joined[f"{column}_second"]
                    ).statistic
                )
            )
            target_counts.append(len(joined))
    if not correlations:
        return {
            "median_spearman": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_splits": 0.0,
            "median_targets": float("nan"),
        }
    values = np.asarray(correlations, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "median_spearman": float(np.median(values)) if values.size else float("nan"),
        "ci_low": float(np.percentile(values, 2.5)) if values.size else float("nan"),
        "ci_high": float(np.percentile(values, 97.5)) if values.size else float("nan"),
        "n_splits": float(values.size),
        "median_targets": float(np.median(target_counts)) if target_counts else float("nan"),
    }


def extrapolation_check(
    frame: pd.DataFrame,
    *,
    n_splits: int = 25,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
    seed: int = 17,
    min_responder_coverage: float = 0.8,
) -> dict[str, float]:
    """Test whether components fitted on one animal half predict held-out profiles.

    Animals are split into A and B.  Components fitted on A predict the
    reliability of both half-profiles at their own animal counts.  Under
    independent errors, ``corr(profile_A, profile_B) = sqrt(rel_A * rel_B)``;
    the observed Pearson correlation between the two half-mean profiles is
    therefore an observable check on the extrapolation, not a self-comparison.
    Responders are restricted to those observed in at least
    ``min_responder_coverage`` of the target's animals before both halves are
    scored; otherwise sparse coverage manufactures apparent shared signal.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    cells = cell_summaries(frame, response=response, animal=animal)
    animals = np.array(sorted(cells["_animal"].dropna().unique()))
    if animals.size < 2 * min_animals:
        return {
            "median_abs_error": float("nan"),
            "bias": float("nan"),
            "n_splits": 0.0,
            "median_targets": float("nan"),
            "median_observed_r": float("nan"),
            "median_predicted_r": float("nan"),
            "gate_passed": 0.0,
        }
    groups = {target: group for target, group in cells.groupby("_target", sort=False)}
    rng = np.random.default_rng(seed)
    split_errors: list[float] = []
    split_bias: list[float] = []
    observed_values: list[float] = []
    predicted_values: list[float] = []
    target_counts: list[int] = []
    for _ in range(n_splits):
        shuffled = rng.permutation(animals)
        half_a = set(shuffled[: shuffled.size // 2])
        half_b = set(shuffled[shuffled.size // 2 :])
        errors: list[float] = []
        for _target, group in groups.items():
            pivot = group.pivot_table(index="_animal", columns="_neuron", values="mean", aggfunc="mean")
            coverage = pivot.notna().sum(axis=0) / max(pivot.shape[0], 1)
            keep = coverage >= min_responder_coverage
            if int(keep.sum()) < 3:
                continue
            matched = group[group["_neuron"].isin(pivot.columns[keep])]
            subset_a = matched[matched["_animal"].isin(half_a)]
            subset_b = matched[matched["_animal"].isin(half_b)]
            components_a = _profile_from_cells(
                subset_a,
                min_events=min_events,
                min_animals=min_animals,
                min_responder_coverage=min_responder_coverage,
            )
            components_b = _profile_from_cells(
                subset_b,
                min_events=min_events,
                min_animals=min_animals,
                min_responder_coverage=min_responder_coverage,
            )
            if components_a is None or components_b is None:
                continue
            profile_a = subset_a.groupby("_neuron")["mean"].mean()
            profile_b = subset_b.groupby("_neuron")["mean"].mean()
            joined = pd.concat([profile_a, profile_b], axis=1, join="inner").dropna()
            if len(joined) < 3:
                continue
            if joined.iloc[:, 0].nunique() < 2 or joined.iloc[:, 1].nunique() < 2:
                continue
            correlation = float(stats.pearsonr(joined.iloc[:, 0], joined.iloc[:, 1]).statistic)
            rel_a = components_a.reliability_at(components_a.n_animals)
            rel_b = components_b.reliability_at(components_b.n_animals)
            predicted = float(np.sqrt(max(rel_a * rel_b, 0.0)))
            if not np.isfinite(correlation) or not np.isfinite(predicted):
                continue
            errors.append(correlation - predicted)
            observed_values.append(correlation)
            predicted_values.append(predicted)
        if errors:
            split_errors.append(float(np.mean(np.abs(errors))))
            split_bias.append(float(np.mean(errors)))
            target_counts.append(len(errors))
    if not split_errors:
        return {
            "median_abs_error": float("nan"),
            "bias": float("nan"),
            "n_splits": 0.0,
            "median_targets": float("nan"),
            "median_observed_r": float("nan"),
            "median_predicted_r": float("nan"),
            "gate_passed": 0.0,
        }
    mae = float(np.median(split_errors))
    bias = float(np.median(split_bias))
    passed = float(mae <= 0.10 and abs(bias) <= 0.05)
    return {
        "median_abs_error": mae,
        "bias": bias,
        "n_splits": float(len(split_errors)),
        "median_targets": float(np.median(target_counts)),
        "median_observed_r": float(np.median(observed_values)),
        "median_predicted_r": float(np.median(predicted_values)),
        "gate_passed": passed,
    }


__all__ = [
    "SUMMARY_COLUMNS",
    "animal_half_replication",
    "animal_jackknife",
    "extrapolation_check",
    "summary_sensitivity",
]
