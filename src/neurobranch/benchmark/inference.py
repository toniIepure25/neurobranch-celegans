"""Animal-cluster inference for paired benchmark comparisons.

Fold-level spread is not an animal-generalisation interval because folds share
animals.  This module resamples animals, recomputes the paired pair-level
Spearman difference, and applies a pre-declared multiplicity correction.  It
never upgrades a descriptive leaderboard entry to a ranking claim on its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neurobranch.benchmark.metrics import spearman

DEFAULT_BASELINE = "cro_static"
DEFAULT_SESOI = 0.05


def _animal_pairs(frame: pd.DataFrame, *, min_events: int = 2) -> pd.DataFrame:
    """Pair means within an animal, so animal is the resampling cluster."""
    grouped = (
        frame.groupby(["dataset_id", "target", "neuron"], dropna=False)
        .agg(
            y_true=("y_true", "mean"),
            y_pred=("y_pred", "mean"),
            n_events=("y_true", "size"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n_events"] >= min_events].reset_index(drop=True)
    return grouped.rename(columns={"y_pred": "y_pred_model"})


def _baseline_pairs(frame: pd.DataFrame, *, min_events: int = 2) -> pd.DataFrame:
    grouped = (
        frame.groupby(["dataset_id", "target", "neuron"], dropna=False)
        .agg(
            y_true=("y_true", "mean"),
            y_pred=("y_pred", "mean"),
            n_events=("y_true", "size"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n_events"] >= min_events].reset_index(drop=True)
    return grouped.rename(columns={"y_pred": "y_pred_baseline", "y_true": "y_true_baseline"})


EVENT_KEYS = ["dataset_id", "target", "neuron", "event_id"]


def _event_rows(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Return one row per exact event key, failing closed on duplicates."""
    missing = [column for column in EVENT_KEYS + ["y_true", "y_pred"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} predictions are missing event-key columns {missing}")
    work = frame[EVENT_KEYS + ["y_true", "y_pred"]].copy()
    duplicated = work.duplicated(EVENT_KEYS, keep=False)
    if bool(duplicated.any()):
        examples = work.loc[duplicated, EVENT_KEYS].head(3).to_dict("records")
        raise ValueError(
            f"{label} has {int(duplicated.sum())} duplicate event-key rows; "
            f"examples: {examples}"
        )
    work = work.dropna(subset=EVENT_KEYS + ["y_true", "y_pred"])
    return work.rename(columns={"y_true": "y_true_event", "y_pred": "y_pred_event"})


def _event_pairs(
    frame: pd.DataFrame,
    *,
    baseline_frame: pd.DataFrame,
    min_events: int,
    require_full_overlap: bool = True,
) -> pd.DataFrame:
    """Join baseline and model on the same event keys, then aggregate by animal."""
    events = _event_rows(frame, label="model")
    merged = events.merge(
        baseline_frame,
        on=EVENT_KEYS,
        how="outer",
        validate="one_to_one",
        suffixes=("_model", "_baseline"),
        indicator=True,
    )
    if require_full_overlap:
        unpaired = merged["_merge"] != "both"
        if bool(unpaired.any()):
            examples = merged.loc[unpaired, EVENT_KEYS].head(3).to_dict("records")
            raise ValueError(
                f"baseline and model event sets differ for {int(unpaired.sum())} event keys; "
                f"examples: {examples}"
            )
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge")
    if merged.empty:
        return merged
    mismatch = (merged["y_true_event_model"] - merged["y_true_event_baseline"]).abs() > 1e-8
    if bool(mismatch.any()):
        raise ValueError(
            "baseline and model y_true are not aligned for "
            f"{int(mismatch.sum())} event-keyed rows; pair on the same events "
            "before computing a delta"
        )
    merged = merged.rename(
        columns={
            "y_true_event_model": "y_true",
            "y_pred_event_model": "y_pred_model",
            "y_pred_event_baseline": "y_pred_baseline",
        }
    )
    grouped = (
        merged.groupby(["dataset_id", "target", "neuron"], dropna=False)
        .agg(
            y_true=("y_true", "mean"),
            y_pred_model=("y_pred_model", "mean"),
            y_pred_baseline=("y_pred_baseline", "mean"),
            n_events=("y_true", "size"),
        )
        .reset_index()
    )
    return grouped[grouped["n_events"] >= min_events].reset_index(drop=True)


def _paired_frame(
    frame: pd.DataFrame,
    *,
    baseline: str,
    min_events: int,
    require_aligned_truth: bool = True,
) -> pd.DataFrame:
    base = frame[frame["model"] == baseline]
    if base.empty:
        raise ValueError(f"baseline {baseline!r} is absent")
    baseline_frame = _event_rows(base, label=f"baseline {baseline!r}")
    rows: list[pd.DataFrame] = []
    for (model, k, strategy), group in frame.groupby(["model", "k", "strategy"], dropna=False):
        if model == baseline:
            continue
        merged = _event_pairs(
            group,
            baseline_frame=baseline_frame,
            min_events=min_events,
            require_full_overlap=require_aligned_truth,
        )
        if merged.empty:
            continue
        merged["model"] = model
        merged["k"] = int(k) if pd.notna(k) else 0
        merged["strategy"] = strategy
        merged["fold"] = group["fold"].iloc[0] if "fold" in group.columns else 0
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def audit_paired_inputs(
    predictions: pd.DataFrame,
    *,
    baseline: str = DEFAULT_BASELINE,
    min_events: int = 2,
) -> dict[str, Any]:
    """Report event-level alignment problems that can invalidate a comparison.

    The unit checked here is the exact event key, not an animal/target/neuron
    aggregate.  Aggregation can hide a baseline/model event-set mismatch.
    """
    required = {"dataset_id", "target", "neuron", "event_id", "model", "y_true", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        return {"status": "missing_columns", "missing": missing}
    event_truth = predictions.groupby(EVENT_KEYS, dropna=False)["y_true"].nunique()
    rows_sharing_event_keys = int(predictions.duplicated(EVENT_KEYS, keep=False).sum())
    base = predictions[predictions["model"] == baseline]
    if base.empty:
        return {"status": "missing_baseline", "baseline": baseline}
    try:
        baseline_frame = _event_rows(base, label=f"baseline {baseline!r}")
    except ValueError as exc:
        return {"status": "duplicate_baseline_keys", "error": str(exc)}
    mismatched = 0
    compared = 0
    duplicate_model_keys = 0
    for (model, _k, _strategy), group in predictions.groupby(["model", "k", "strategy"], dropna=False):
        if model == baseline:
            continue
        duplicate_model_keys += int(group.duplicated(EVENT_KEYS, keep=False).sum())
        try:
            merged = _event_pairs(group, baseline_frame=baseline_frame, min_events=min_events)
        except ValueError as exc:
            return {"status": "unaligned_or_duplicate", "model": model, "error": str(exc)}
        if merged.empty:
            continue
        compared += int(merged["n_events"].sum())
        # _event_pairs would already have raised on a mismatch; this count is a
        # defensive record for the artifact.
        mismatched += 0
    folds = (
        predictions.groupby(["dataset_id"], dropna=False)["fold"].nunique()
        if "fold" in predictions.columns
        else pd.Series(dtype=int)
    )
    return {
        "status": "ok",
        "n_rows": int(len(predictions)),
        "n_event_groups": int(event_truth.size),
        "n_event_groups_with_multiple_truths": int((event_truth > 1).sum()),
        "event_truth_mismatch_fraction": float((event_truth > 1).mean()),
        "n_rows_sharing_event_keys": rows_sharing_event_keys,
        "n_duplicate_model_keys": duplicate_model_keys,
        "n_paired_events_compared": compared,
        "n_paired_y_true_mismatches": mismatched,
        "paired_y_true_mismatch_fraction": (mismatched / compared) if compared else None,
        "max_folds_per_animal": int(folds.max()) if not folds.empty else None,
        "caveat": "Event-keyed audit; a nonzero duplicate count or a join error blocks inference.",
    }


def _bootstrap_delta(
    merged: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    clusters = merged["dataset_id"].astype(str).to_numpy()
    unique = np.unique(clusters)
    if unique.size < 2:
        return np.asarray([], dtype=float)
    index_by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot), dtype=float)
    for boot in range(int(n_boot)):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([index_by_cluster[cluster] for cluster in chosen])
        sample = merged.iloc[indices]
        draws[boot] = spearman(sample["y_true"], sample["y_pred_model"]) - spearman(
            sample["y_true"], sample["y_pred_baseline"]
        )
    return draws[np.isfinite(draws)]


def _jackknife_delta(merged: pd.DataFrame) -> np.ndarray:
    """Leave-one-animal-out deltas for BCa acceleration."""
    clusters = merged["dataset_id"].astype(str).to_numpy()
    unique = np.unique(clusters)
    values: list[float] = []
    for cluster in unique:
        sample = merged.iloc[np.flatnonzero(clusters != cluster)]
        if sample["dataset_id"].nunique() < 2:
            continue
        delta = spearman(sample["y_true"], sample["y_pred_model"]) - spearman(
            sample["y_true"], sample["y_pred_baseline"]
        )
        if np.isfinite(delta):
            values.append(float(delta))
    return np.asarray(values, dtype=float)


def _bca_interval(
    draws: np.ndarray,
    *,
    point: float,
    jackknife: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, str]:
    """Bias-corrected accelerated bootstrap interval.

    The percentile interval undercovered in the Phase 12R synthetic audit
    (0.900--0.917 at nominal 0.95).  BCa is a pre-registered replacement; if its
    correction is numerically undefined, the function returns the percentile
    interval and labels the fallback rather than hiding it.
    """
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    if draws.size < 20 or jackknife.size < 3 or not np.isfinite(point):
        return float(low), float(high), "percentile_fallback"
    proportion = float(np.mean(draws < point))
    if proportion <= 0.0 or proportion >= 1.0:
        return float(low), float(high), "percentile_fallback"
    z0 = float(stats.norm.ppf(proportion))
    mean_jack = float(np.mean(jackknife))
    deviations = mean_jack - jackknife
    denominator = 6.0 * float(np.sum(deviations**2)) ** 1.5
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float(low), float(high), "percentile_fallback"
    acceleration = float(np.sum(deviations**3) / denominator)
    z_low = float(stats.norm.ppf(alpha / 2.0))
    z_high = float(stats.norm.ppf(1.0 - alpha / 2.0))

    def adjusted(z: float) -> float:
        return float(stats.norm.cdf(z0 + (z0 + z) / (1.0 - acceleration * (z0 + z))))

    low_q = adjusted(z_low)
    high_q = adjusted(z_high)
    if not (np.isfinite(low_q) and np.isfinite(high_q) and 0.0 <= low_q < high_q <= 1.0):
        return float(low), float(high), "percentile_fallback"
    return float(np.quantile(draws, low_q)), float(np.quantile(draws, high_q)), "bca"


def paired_cluster_inference(
    predictions: pd.DataFrame,
    *,
    baseline: str = DEFAULT_BASELINE,
    n_boot: int = 500,
    seed: int = 29,
    min_events: int = 2,
    sesoi: float = DEFAULT_SESOI,
    require_aligned_truth: bool = True,
) -> pd.DataFrame:
    """Cluster-bootstrap paired model deltas with Holm adjustment and MDE.

    The function fails closed when the baseline and model rows do not share the
    same event truth.  Use ``require_aligned_truth=False`` only to reproduce a
    historical artifact, never for a new claim.
    """
    if n_boot < 50:
        raise ValueError("n_boot must be >= 50")
    merged_all = _paired_frame(
        predictions,
        baseline=baseline,
        min_events=min_events,
        require_aligned_truth=require_aligned_truth,
    )
    if merged_all.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for config_index, ((model, k, strategy), group) in enumerate(
        merged_all.groupby(["model", "k", "strategy"], dropna=False)
    ):
        draws = _bootstrap_delta(group, n_boot=n_boot, seed=seed + config_index)
        point = spearman(group["y_true"], group["y_pred_model"]) - spearman(
            group["y_true"], group["y_pred_baseline"]
        )
        if draws.size < 10:
            rows.append(
                {
                    "model": model,
                    "k": int(k),
                    "strategy": strategy,
                    "delta": float(point) if np.isfinite(point) else float("nan"),
                    "delta_ci_low": float("nan"),
                    "delta_ci_high": float("nan"),
                    "p_value": float("nan"),
                    "p_holm": float("nan"),
                    "n_animal_clusters": int(group["dataset_id"].nunique()),
                    "mde_80": float("nan"),
                    "status": "insufficient_clusters",
                }
            )
            continue
        low, high, interval_method = _bca_interval(
            draws,
            point=float(point),
            jackknife=_jackknife_delta(group),
        )
        standard_error = float(np.std(draws, ddof=1))
        p_value = (
            2.0 * float(stats.norm.sf(abs(float(point)) / standard_error))
            if standard_error > 0.0
            else float("nan")
        )
        p_value = min(1.0, p_value + 1.0 / (draws.size + 1.0))
        mde = 2.802 * standard_error
        rows.append(
            {
                "model": model,
                "k": int(k),
                "strategy": strategy,
                "delta": float(point) if np.isfinite(point) else float("nan"),
                "delta_ci_low": float(low),
                "delta_ci_high": float(high),
                "interval_method": interval_method,
                "p_value": float(p_value),
                "p_holm": float("nan"),
                "p_method": "normal_bootstrap_se",
                "n_animal_clusters": int(group["dataset_id"].nunique()),
                "mde_80": float(mde),
                "status": "descriptive",
            }
        )
    frame = pd.DataFrame(rows)
    valid = frame["p_value"].notna()
    if valid.any():
        p = frame.loc[valid, "p_value"].to_numpy(dtype=float)
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        running = 0.0
        m = p.size
        for rank, index in enumerate(order):
            value = min(1.0, p[index] * (m - rank))
            running = max(running, value)
            adjusted[index] = running
        frame.loc[valid, "p_holm"] = adjusted
    for index, row in frame.iterrows():
        if row["status"] != "descriptive":
            continue
        excludes_zero = (row["delta_ci_low"] > 0) or (row["delta_ci_high"] < 0)
        if bool(excludes_zero and row["p_holm"] < 0.05 and row["mde_80"] <= sesoi):
            frame.at[index, "status"] = "supported"
        elif row["mde_80"] > sesoi:
            frame.at[index, "status"] = "underpowered"
        else:
            frame.at[index, "status"] = "inconclusive"
    return frame.sort_values(
        ["delta", "model", "k", "strategy"], ascending=[False, True, True, True]
    ).reset_index(drop=True)


def inference_summary(frame: pd.DataFrame, *, sesoi: float = DEFAULT_SESOI) -> dict[str, Any]:
    """JSON-friendly summary that never overstates a positive result."""
    if frame.empty:
        return {"status": "no_configurations", "sesoi": sesoi}
    supported = frame[frame["status"] == "supported"]
    return {
        "status": "ok",
        "unit": "animal_cluster",
        "n_configurations": int(len(frame)),
        "n_supported": int(len(supported)),
        "sesoi": float(sesoi),
        "best_supported_delta": float(supported["delta"].max()) if len(supported) else None,
        "caveat": "Supported means the paired animal-cluster interval excludes zero and MDE <= SESOI.",
    }


__all__ = [
    "DEFAULT_BASELINE",
    "DEFAULT_SESOI",
    "audit_paired_inputs",
    "inference_summary",
    "paired_cluster_inference",
]
