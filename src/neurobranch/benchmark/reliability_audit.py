"""Phase 15 reliability audit: estimands, cluster uncertainty and null controls.

This module deliberately contains no measurement model.  It operates on the
moment components already used by the Phase 11 estimator and asks what the
observed animal counts identify, what is extrapolated, and whether the
uncertainty and null behaviour are adequate.

The primary uncertainty estimate is the delete-one-animal jackknife in
:mod:`neurobranch.benchmark.robustness`.  The Bayesian bootstrap here is a
cross-check that avoids duplicating animal rows; the two-way bootstrap in
:mod:`neurobranch.uncertainty` is retained only as an invalid-interval
diagnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neurobranch.benchmark.robustness import (
    _estimand_reliability,
    _jackknife_from_matrices,
    _TargetMatrix,
)
from neurobranch.benchmark.variance import (
    DEFAULT_ANIMAL_COLUMN,
    _components_from_arrays,
    cell_summaries,
    profile_components_from_cells,
    summarise_reliability_estimands,
)

ESTIMANDS = ("obs", "fold", "id")


def _selected(matrices: list[_TargetMatrix], estimand: str) -> list[_TargetMatrix]:
    if estimand not in ESTIMANDS:
        raise ValueError(f"estimand must be one of {ESTIMANDS}")
    if estimand == "id":
        return [matrix for matrix in matrices if matrix.in_identified_subset]
    return list(matrices)


def _n_eval(matrix: _TargetMatrix, estimand: str, fold_size: int) -> int:
    return matrix.n_animals if estimand in {"obs", "id"} else fold_size


def _aggregate_matrix_values(
    matrices: list[_TargetMatrix],
    values: list[float],
) -> float:
    return float(np.nanmean(np.asarray(values, dtype=float))) if values else float("nan")


def _weighted_components(
    values: np.ndarray,
    noise: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float] | None:
    """Weighted analogue of the additive moment estimator.

    Weights are animal weights summing to one.  Missing cells are handled with
    per-responder normalisation.  The interaction correction uses the effective
    animal sample size implied by the weights.  This is a cross-check, not a
    replacement for the primary unweighted estimator.
    """
    present = np.isfinite(values)
    if values.shape[1] < 3 or not np.any(present):
        return None
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    denominator = np.nansum(weights[:, None] * present, axis=0)
    numerator = np.nansum(weights[:, None] * values, axis=0)
    means = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    if np.isfinite(means).sum() < 3:
        return None
    q_per_responder = np.divide(
        np.nansum((weights[:, None] ** 2) * present, axis=0),
        denominator**2,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    observed_var = float(np.nanvar(means, ddof=1))
    sampling = np.divide(
        np.nansum((weights[:, None] ** 2) * noise, axis=0),
        denominator**2,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    sampling_mean = float(np.nanmean(sampling))
    q_mean = float(np.nanmean(q_per_responder))
    if not np.isfinite(observed_var) or observed_var <= 0 or not np.isfinite(sampling_mean) or not np.isfinite(q_mean):
        return None

    centered = values - means
    weighted_var = np.divide(
        np.nansum(weights[:, None] * centered**2, axis=0),
        1.0 - q_per_responder,
        out=np.full_like(denominator, np.nan),
        where=(1.0 - q_per_responder) > 1e-12,
    )
    individual_noise = np.divide(
        np.nansum(weights[:, None] * noise, axis=0),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    interaction_r = np.maximum(weighted_var - individual_noise, 0.0)
    interaction_mean = float(np.nanmean(interaction_r))
    if not np.isfinite(interaction_mean):
        return None
    signal = max(observed_var - sampling_mean - interaction_mean * q_mean, 0.0)
    return float(signal), float(sampling_mean), float(interaction_mean)


def bayesian_bootstrap(
    matrices: list[_TargetMatrix],
    *,
    estimand: str = "fold",
    fold_size: int = 22,
    n_draws: int = 2_000,
    seed: int = 15,
) -> np.ndarray:
    """Dirichlet-weighted animal bootstrap for an aggregate reliability estimand.

    One global weight vector is drawn per replicate and normalised within each
    target.  This preserves cross-target animal dependence and, unlike the
    two-way row bootstrap, never duplicates an animal row.
    """
    if n_draws < 20:
        raise ValueError("n_draws must be at least 20")
    selected = _selected(matrices, estimand)
    animals = sorted({animal for matrix in selected for animal in matrix.animal_labels})
    if not selected or len(animals) < 3:
        return np.full(n_draws, np.nan)
    animal_index = {name: index for index, name in enumerate(animals)}
    rng = np.random.default_rng(seed)
    draws = np.full(n_draws, np.nan)
    for draw in range(n_draws):
        weights = rng.dirichlet(np.ones(len(animals)))
        values: list[float] = []
        for matrix in selected:
            local_weights = np.array([weights[animal_index[name]] for name in matrix.animal_labels], dtype=float)
            if not np.isfinite(local_weights).all() or local_weights.sum() <= 0:
                continue
            components = _weighted_components(matrix.values, matrix.noise, local_weights)
            if components is None:
                continue
            signal, noise, interaction = components
            values.append(
                float(_estimand_reliability(signal, noise, interaction, _n_eval(matrix, estimand, fold_size)))
            )
        draws[draw] = _aggregate_matrix_values(selected, values)
    return draws


def _null_draw(
    matrices: list[_TargetMatrix],
    *,
    estimand: str,
    fold_size: int,
    mode: str,
    rng: np.random.Generator,
) -> float:
    selected = _selected(matrices, estimand)
    values: list[float] = []
    for matrix in selected:
        transformed = matrix.values.copy()
        if mode == "signflip":
            signs = rng.choice(np.array([-1.0, 1.0]), size=transformed.shape[0])
            transformed = transformed * signs[:, None]
        elif mode == "permutation":
            for row in range(transformed.shape[0]):
                finite = np.isfinite(transformed[row])
                if finite.sum() > 1:
                    transformed[row, finite] = rng.permutation(transformed[row, finite])
        else:
            raise ValueError("mode must be 'signflip' or 'permutation'")
        component = _components_from_arrays(transformed, matrix.noise, min_animals=3)
        if component is None:
            continue
        values.append(
            float(
                _estimand_reliability(
                    component.signal,
                    component.noise,
                    component.interaction,
                    _n_eval(matrix, estimand, fold_size),
                )
            )
        )
    return _aggregate_matrix_values(selected, values)


def null_distributions(
    matrices: list[_TargetMatrix],
    *,
    estimand: str = "fold",
    fold_size: int = 22,
    n_draws: int = 1_000,
    seed: int = 15,
) -> dict[str, dict[str, float]]:
    """Responder-permutation and animal-sign-flip null distributions."""
    if n_draws < 20:
        raise ValueError("n_draws must be at least 20")
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for mode in ("permutation", "signflip"):
        draws = np.array(
            [
                _null_draw(matrices, estimand=estimand, fold_size=fold_size, mode=mode, rng=rng)
                for _ in range(n_draws)
            ],
            dtype=float,
        )
        finite = draws[np.isfinite(draws)]
        out[mode] = {
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "q95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
            "n_valid": float(finite.size),
        }
    return out


def audit_estimands(
    matrices: list[_TargetMatrix],
    *,
    fold_size: int = 22,
) -> dict[str, Any]:
    """Estimates, jackknife intervals and target tables for all estimands."""
    table_rows: list[dict[str, Any]] = []
    for matrix in matrices:
        observed = float(
            _estimand_reliability(matrix.signal, matrix.noise_mean, matrix.interaction_mean, matrix.n_animals)
        )
        at_fold = float(
            _estimand_reliability(matrix.signal, matrix.noise_mean, matrix.interaction_mean, fold_size)
        )
        table_rows.append(
            {
                "target": matrix.target,
                "n_animals": matrix.n_animals,
                "n_responders": matrix.n_responders,
                "signal": matrix.signal,
                "noise": matrix.noise_mean,
                "interaction": matrix.interaction_mean,
                "reliability_obs": observed,
                "ceiling_obs": float(np.sqrt(observed)),
                "reliability_22": at_fold,
                "ceiling_22": float(np.sqrt(at_fold)),
                "reliability_id": observed if matrix.in_identified_subset else float("nan"),
                "ceiling_id": float(np.sqrt(observed)) if matrix.in_identified_subset else float("nan"),
                "in_identified_subset": matrix.in_identified_subset,
                "extrapolated_22": not matrix.in_identified_subset,
            }
        )
    table = pd.DataFrame(table_rows)
    summary = summarise_reliability_estimands(table.rename(columns={"n_animals": "n_animals"}), fold_size=fold_size)
    jackknife: dict[str, dict[str, Any]] = {}
    for estimand in ESTIMANDS:
        jackknife[estimand] = _jackknife_from_matrices(
            matrices,
            estimand=estimand,
            fold_size=fold_size,
            level=0.95,
        )
    return {"table": table, "summary": summary, "jackknife": jackknife}


def _truth_value(matrix: _TargetMatrix, estimand: str, fold_size: int) -> float:
    n_eval = matrix.n_animals if estimand in {"obs", "id"} else fold_size
    return float(_estimand_reliability(matrix.signal, matrix.noise_mean, matrix.interaction_mean, n_eval))


def _truth_aggregate(matrices: list[_TargetMatrix], estimand: str, fold_size: int) -> float:
    selected = _selected(matrices, estimand)
    return float(np.mean([_truth_value(matrix, estimand, fold_size) for matrix in selected])) if selected else float("nan")


def _simulate_matrix(matrix: _TargetMatrix, rng: np.random.Generator) -> _TargetMatrix:
    """Draw one additive animal x responder data set with fixed missingness."""
    shape = matrix.values.shape
    signal_sd = float(np.sqrt(max(matrix.signal, 0.0)))
    interaction_sd = float(np.sqrt(max(matrix.interaction_mean, 0.0)))
    means = np.asarray(rng.normal(0.0, signal_sd, size=shape[1]), dtype=float)
    interaction = np.asarray(rng.normal(0.0, interaction_sd, size=shape), dtype=float)
    sampling = np.asarray(
        rng.normal(0.0, np.sqrt(np.maximum(matrix.noise, 0.0)), size=shape), dtype=float
    )
    values = means[None, :] + interaction + sampling
    values = np.where(np.isfinite(matrix.values), values, np.nan)
    component = _components_from_arrays(values, matrix.noise, min_animals=3)
    if component is None:
        return _TargetMatrix(
            target=matrix.target,
            animal_labels=matrix.animal_labels,
            values=values,
            noise=matrix.noise,
            signal=0.0,
            noise_mean=float(np.nanmean(matrix.noise)),
            interaction_mean=0.0,
            n_animals=matrix.n_animals,
            n_responders=matrix.n_responders,
            in_identified_subset=matrix.in_identified_subset,
        )
    return _TargetMatrix(
        target=matrix.target,
        animal_labels=matrix.animal_labels,
        values=values,
        noise=matrix.noise,
        signal=component.signal,
        noise_mean=component.noise,
        interaction_mean=component.interaction,
        n_animals=matrix.n_animals,
        n_responders=matrix.n_responders,
        in_identified_subset=matrix.in_identified_subset,
    )


def calibration_study(
    matrices: list[_TargetMatrix],
    *,
    fold_size: int = 22,
    n_reps: int = 1_000,
    seed: int = 15,
) -> dict[str, Any]:
    """Parametric coverage study for the jackknife intervals.

    The simulation uses the real target sizes, missingness masks and sampling
    variances, and draws new signal, interaction and noise values.  The truth is
    the same functional evaluated with the generating components.  No threshold
    is chosen from the result.
    """
    if n_reps < 20:
        raise ValueError("n_reps must be at least 20")
    rng = np.random.default_rng(seed)
    truth = {estimand: _truth_aggregate(matrices, estimand, fold_size) for estimand in ESTIMANDS}
    records: dict[str, list[dict[str, Any]]] = {estimand: [] for estimand in ESTIMANDS}
    for _ in range(n_reps):
        simulated = [_simulate_matrix(matrix, rng) for matrix in matrices]
        for estimand in ESTIMANDS:
            result = _jackknife_from_matrices(simulated, estimand=estimand, fold_size=fold_size, level=0.95)
            records[estimand].append(
                {
                    "estimate": result["estimate"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "status": result["status"],
                }
            )
    summary: dict[str, Any] = {"n_reps": float(n_reps), "fold_size": float(fold_size), "truth": truth, "estimands": {}}
    for estimand in ESTIMANDS:
        frame = pd.DataFrame(records[estimand])
        valid = frame[frame["status"] == "ok"].dropna()
        if valid.empty:
            summary["estimands"][estimand] = {
                "coverage": float("nan"),
                "bias": float("nan"),
                "mean_width": float("nan"),
                "n_valid": 0.0,
                "n_total": float(n_reps),
                "status": "no_valid_replicates",
            }
            continue
        covered = (valid["ci_low"] <= truth[estimand]) & (truth[estimand] <= valid["ci_high"])
        coverage = float(covered.mean())
        summary["estimands"][estimand] = {
            "coverage": coverage,
            "bias": float((valid["estimate"] - truth[estimand]).mean()),
            "mean_width": float((valid["ci_high"] - valid["ci_low"]).mean()),
            "n_valid": float(len(valid)),
            "n_total": float(n_reps),
            "status": "calibrated" if 0.93 <= coverage <= 0.97 else "not_calibrated",
        }
    return summary


def ranking_stability(
    frame: pd.DataFrame,
    *,
    n_splits: int = 15,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 3,
    min_responder_coverage: float = 0.8,
    top_k: int = 10,
    seed: int = 15,
    one_event_variance: float | None = None,
    event_variance: float | None = None,
) -> dict[str, Any]:
    """Split animals in half and compare target ranks of observed-count reliability.

    The ranks are computed from ``R_obs`` in each half; the statistic is the
    Spearman correlation on the intersection of eligible targets and the Jaccard
    overlap of the top-k sets.  Half-split eligibility uses ``min_animals=3``
    because each half has roughly half the animals; the target intersection is
    reported with every split.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    cells = cell_summaries(
        frame,
        response=response,
        animal=animal,
        one_event_variance=one_event_variance,
        event_variance=event_variance,
    )
    animals = np.array(sorted(cells["_animal"].dropna().unique()))
    if animals.size < 6:
        return {
            "median_spearman": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "median_jaccard": float("nan"),
            "n_splits": 0.0,
            "median_targets": float("nan"),
            "top_k": float(top_k),
            "status": "insufficient_animals",
        }
    rng = np.random.default_rng(seed)
    correlations: list[float] = []
    jaccards: list[float] = []
    target_counts: list[int] = []
    for _ in range(n_splits):
        shuffled = rng.permutation(animals)
        halves = (set(shuffled[: shuffled.size // 2]), set(shuffled[shuffled.size // 2 :]))
        frames: list[pd.DataFrame] = []
        for half in halves:
            components = profile_components_from_cells(
                cells[cells["_animal"].isin(half)],
                min_events=min_events,
                min_animals=min_animals,
                min_responder_coverage=min_responder_coverage,
            )
            if components.empty:
                frames.append(pd.DataFrame(columns=["target", "reliability_obs"]))
                continue
            table = components.copy()
            signal = table["signal"].to_numpy(dtype=float)
            error = (table["noise"] + table["interaction"]).to_numpy(dtype=float)
            n_animals = table["n_animals"].to_numpy(dtype=float)
            denominator = signal + error / n_animals
            reliability = np.divide(
                signal,
                denominator,
                out=np.zeros_like(signal),
                where=denominator > 0,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "target": table["target"].to_numpy(),
                        "reliability_obs": np.clip(reliability, 0.0, 1.0),
                    }
                )
            )
        joined = frames[0].merge(frames[1], on="target", suffixes=("_first", "_second")).dropna()
        if len(joined) >= 3:
            correlations.append(float(stats.spearmanr(joined["reliability_obs_first"], joined["reliability_obs_second"]).statistic))
            first_top = set(joined.nlargest(min(top_k, len(joined)), "reliability_obs_first")["target"])
            second_top = set(joined.nlargest(min(top_k, len(joined)), "reliability_obs_second")["target"])
            union = first_top | second_top
            jaccards.append(float(len(first_top & second_top) / len(union)) if union else float("nan"))
            target_counts.append(len(joined))
    finite_correlations = np.asarray([value for value in correlations if np.isfinite(value)], dtype=float)
    finite_jaccards = np.asarray([value for value in jaccards if np.isfinite(value)], dtype=float)
    if finite_correlations.size < 3:
        return {
            "median_spearman": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "median_jaccard": float("nan"),
            "n_splits": float(finite_correlations.size),
            "median_targets": float(np.median(target_counts)) if target_counts else float("nan"),
            "top_k": float(top_k),
            "status": "insufficient_splits",
        }
    ci_low = float(np.percentile(finite_correlations, 2.5))
    median_jaccard = float(np.median(finite_jaccards)) if finite_jaccards.size else float("nan")
    return {
        "median_spearman": float(np.median(finite_correlations)),
        "ci_low": ci_low,
        "ci_high": float(np.percentile(finite_correlations, 97.5)),
        "median_jaccard": median_jaccard,
        "n_splits": float(finite_correlations.size),
        "median_targets": float(np.median(target_counts)) if target_counts else float("nan"),
        "top_k": float(top_k),
        "status": "ok" if ci_low > 0.50 and np.isfinite(median_jaccard) and median_jaccard > 0.60 else "no_ranking",
    }


__all__ = [
    "ESTIMANDS",
    "audit_estimands",
    "bayesian_bootstrap",
    "calibration_study",
    "null_distributions",
    "ranking_stability",
]
