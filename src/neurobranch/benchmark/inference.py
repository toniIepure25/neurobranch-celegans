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
    baseline_pairs = _baseline_pairs(base, min_events=min_events)
    rows: list[pd.DataFrame] = []
    for (model, k, strategy), group in frame.groupby(["model", "k", "strategy"], dropna=False):
        if model == baseline:
            continue
        pairs = _animal_pairs(group, min_events=min_events)
        merged = pairs.merge(baseline_pairs, on=["dataset_id", "target", "neuron"], how="inner")
        if merged.empty:
            continue
        merged = merged.dropna(subset=["y_true", "y_pred_model", "y_pred_baseline", "dataset_id"])
        if require_aligned_truth:
            mismatch = (merged["y_true"] - merged["y_true_baseline"]).abs() > 1e-8
            if bool(mismatch.any()):
                raise ValueError(
                    "baseline and model y_true are not aligned for "
                    f"{int(mismatch.sum())} paired rows in {model!r}; "
                    "pair on the same events before computing a delta"
                )
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
    """Report alignment problems that can invalidate a paired comparison.

    This is diagnostic: it does not silently drop rows.  The benchmark claim is
    only interpretable when ``y_true`` agrees between baseline and model for the
    same event and the resampling unit is unambiguous.
    """
    required = {"dataset_id", "target", "neuron", "event_id", "model", "y_true", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        return {"status": "missing_columns", "missing": missing}
    keys = ["dataset_id", "target", "neuron", "event_id"]
    event_truth = predictions.groupby(keys, dropna=False)["y_true"].nunique()
    rows_sharing_event_keys = int(predictions.duplicated(keys, keep=False).sum())
    base = predictions[predictions["model"] == baseline]
    base_pairs = _baseline_pairs(base, min_events=min_events)
    mismatched = 0
    compared = 0
    for (model, _k, _strategy), group in predictions.groupby(["model", "k", "strategy"], dropna=False):
        if model == baseline:
            continue
        pairs = _animal_pairs(group, min_events=min_events)
        merged = pairs.merge(base_pairs, on=["dataset_id", "target", "neuron"], how="inner")
        if merged.empty:
            continue
        compared += int(len(merged))
        mismatched += int((np.abs(merged["y_true"] - merged["y_true_baseline"]) > 1e-8).sum())
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
        "n_paired_rows_compared": compared,
        "n_paired_y_true_mismatches": mismatched,
        "paired_y_true_mismatch_fraction": (mismatched / compared) if compared else None,
        "max_folds_per_animal": int(folds.max()) if not folds.empty else None,
        "caveat": "Diagnostic only; paired inference is not repaired by this audit.",
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
            sample["y_true_baseline"], sample["y_pred_baseline"]
        )
    return draws[np.isfinite(draws)]


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
            group["y_true_baseline"], group["y_pred_baseline"]
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
        low, high = np.quantile(draws, [0.025, 0.975])
        p_value = 2.0 * min(float(np.mean(draws <= 0.0)), float(np.mean(draws >= 0.0)))
        p_value = min(1.0, p_value + 1.0 / (draws.size + 1.0))
        mde = 2.802 * float(np.std(draws, ddof=1))
        rows.append(
            {
                "model": model,
                "k": int(k),
                "strategy": strategy,
                "delta": float(point) if np.isfinite(point) else float("nan"),
                "delta_ci_low": float(low),
                "delta_ci_high": float(high),
                "p_value": float(p_value),
                "p_holm": float("nan"),
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
