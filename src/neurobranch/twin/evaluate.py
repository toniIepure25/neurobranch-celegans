"""Metrics and paired uncertainty for the Phase 8 adaptive twin."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neurobranch.benchmark.metrics import (
    interval_score,
    pair_spearman_from_pairs,
    pair_table,
    spearman,
)


def trajectory_pearson(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean per-event Pearson correlation of post-stimulus trajectories."""
    a = np.asarray(true, dtype=float)
    b = np.asarray(pred, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("trajectory arrays must have the same 2-D shape")
    vals = []
    for i in range(a.shape[0]):
        x = a[i]
        y = b[i]
        ok = np.isfinite(x) & np.isfinite(y)
        if mask is not None:
            ok &= np.asarray(mask)[i].astype(bool)
        if ok.sum() < 3 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
            continue
        vals.append(float(stats.pearsonr(x[ok], y[ok]).statistic))
    return float(np.mean(vals)) if vals else float("nan")


def frame_metrics(frame: pd.DataFrame, *, min_events: int = 1) -> dict[str, float]:
    """Scalar/profile metrics for one prediction frame."""
    needed = {"target", "neuron", "y_true", "y_pred"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"prediction frame missing {sorted(missing)}")
    clean = frame[np.isfinite(frame["y_true"]) & np.isfinite(frame["y_pred"])].copy()
    if clean.empty:
        return {
            "pair_spearman": float("nan"),
            "profile_spearman": float("nan"),
            "coverage": float("nan"),
            "interval_score": float("nan"),
            "n_events": 0.0,
            "n_pairs": 0.0,
        }
    pairs = pair_table(clean, min_events=min_events)
    pair_r = pair_spearman_from_pairs(pairs) if not pairs.empty else float("nan")
    profiles = (
        clean.groupby("target")
        .apply(
            lambda g: spearman(g["y_true"], g["y_pred"]),
            include_groups=False,
        )
        .dropna()
    )
    profile_r = float(profiles.mean()) if not profiles.empty else float("nan")
    coverage = float("nan")
    score = float("nan")
    if {"interval_low", "interval_high"}.issubset(clean.columns):
        coverage = float(
            np.mean((clean["y_true"] >= clean["interval_low"]) & (clean["y_true"] <= clean["interval_high"]))
        )
        score = float(
            interval_score(
                clean.rename(columns={"interval_low": "lower", "interval_high": "upper"}),
                lower="lower",
                upper="upper",
            )
        )
    return {
        "pair_spearman": float(pair_r),
        "profile_spearman": float(profile_r),
        "coverage": coverage,
        "interval_score": score,
        "n_events": float(len(clean)),
        "n_pairs": float(len(pairs)),
    }


def add_intervals(frame: pd.DataFrame, *, df: float = 5.0, scale: float = 1.0) -> pd.DataFrame:
    out = frame.copy()
    q = float(stats.t.ppf(0.95, df))
    out["interval_low"] = out["y_pred"] - q * out["sigma"] * scale
    out["interval_high"] = out["y_pred"] + q * out["sigma"] * scale
    return out


def calibration_scale(frame: pd.DataFrame, *, df: float = 5.0) -> float:
    """Validation-only conformal-like scale for 90% predictive intervals."""
    if frame.empty:
        return 1.0
    q = float(stats.t.ppf(0.95, df))
    denom = np.maximum(np.abs(frame["sigma"].to_numpy(dtype=float)) * q, 1e-6)
    ratio = np.abs(frame["y_true"].to_numpy(dtype=float) - frame["y_pred"].to_numpy(dtype=float)) / denom
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return 1.0
    return float(np.quantile(ratio, 0.90))


def paired_metric_delta(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    metric: str = "pair_spearman",
    n_boot: int = 1000,
    seed: int = 17,
) -> dict[str, float]:
    """Animal-cluster bootstrap of a paired metric difference.

    ``frame_a`` and ``frame_b`` must contain predictions on the same test rows.
    Rows are merged on event/target/responder and resampled by animal.
    """
    key = ["event_id", "target", "neuron"]
    for col in key + ["y_true", "y_pred", "dataset_id"]:
        if col not in frame_a.columns or col not in frame_b.columns:
            raise ValueError(f"paired frames missing {col!r}")
    a = frame_a[key + ["y_true", "y_pred", "dataset_id"]].rename(columns={"y_pred": "pred_a"})
    b = frame_b[key + ["y_pred"]].rename(columns={"y_pred": "pred_b"})
    merged = a.merge(b, on=key, how="inner")
    if merged.empty:
        return {
            "delta": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "n_boot": 0.0,
            "p_one_sided": float("nan"),
        }
    merged = merged.rename(columns={"y_true": "y"})
    merged = merged.reset_index(drop=True)
    animal_values = merged["dataset_id"].astype(str).to_numpy()
    animals = sorted(np.unique(animal_values))
    rng = np.random.default_rng(seed)
    values = []
    # For the pair-level metric, pre-aggregate to animal x pair sums once.
    # Rebuilding the full pair table for every row bootstrap dominated wall time.
    pair_sums: pd.DataFrame | None = None
    if metric == "pair_spearman":
        pair_sums = (
            merged.groupby(["dataset_id", "target", "neuron"], sort=False)
            .agg(
                y_sum=("y", "sum"),
                y_n=("y", "size"),
                a_sum=("pred_a", "sum"),
                b_sum=("pred_b", "sum"),
            )
            .reset_index()
        )
        group_values = pair_sums["dataset_id"].astype(str).to_numpy()
        positions = {a: np.flatnonzero(group_values == a) for a in animals}
    else:
        positions = {a: np.flatnonzero(animal_values == a) for a in animals}
    for _ in range(n_boot):
        sampled = rng.choice(animals, size=len(animals), replace=True)
        if metric == "pair_spearman" and pair_sums is not None:
            boot_pairs = pair_sums.iloc[np.concatenate([positions[a] for a in sampled])]
            agg = boot_pairs.groupby(["target", "neuron"], sort=False)[
                ["y_sum", "y_n", "a_sum", "b_sum"]
            ].sum()
            target_mean = agg["y_sum"] / agg["y_n"]
            va = float(target_mean.corr(agg["a_sum"] / agg["y_n"], method="spearman"))
            vb = float(target_mean.corr(agg["b_sum"] / agg["y_n"], method="spearman"))
        else:
            boot = merged.iloc[np.concatenate([positions[a] for a in sampled])].reset_index(drop=True)
            if metric == "pair_spearman":
                pa = pair_table(
                    boot.rename(columns={"pred_a": "y_pred", "y": "y_true"}), min_events=1
                )
                pb = pair_table(
                    boot.rename(columns={"pred_b": "y_pred", "y": "y_true"}), min_events=1
                )
                va = pair_spearman_from_pairs(pa) if not pa.empty else float("nan")
                vb = pair_spearman_from_pairs(pb) if not pb.empty else float("nan")
            else:
                va = frame_metrics(boot.rename(columns={"pred_a": "y_pred", "y": "y_true"}))[metric]
                vb = frame_metrics(boot.rename(columns={"pred_b": "y_pred", "y": "y_true"}))[metric]
        if np.isfinite(va) and np.isfinite(vb):
            values.append(float(vb - va))
    point_a = frame_metrics(merged.rename(columns={"pred_a": "y_pred", "y": "y_true"}))[metric]
    point_b = frame_metrics(merged.rename(columns={"pred_b": "y_pred", "y": "y_true"}))[metric]
    if not values:
        return {
            "delta": float(point_b - point_a),
            "low": float("nan"),
            "high": float("nan"),
            "n_boot": 0.0,
            "p_one_sided": float("nan"),
        }
    values_arr = np.asarray(values, dtype=float)
    return {
        "delta": float(point_b - point_a),
        "low": float(np.percentile(values_arr, 2.5)),
        "high": float(np.percentile(values_arr, 97.5)),
        "n_boot": float(len(values)),
        "p_one_sided": float(np.mean(values_arr <= 0.0)),
    }


def per_fold_metric(frame: pd.DataFrame, *, metric: str = "pair_spearman") -> float:
    return float(frame_metrics(frame)[metric])


def metric_by_block(
    frame: pd.DataFrame,
    *,
    block_cols: tuple[str, ...] = ("fold", "model_seed", "strategy", "k"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, g in frame.groupby(list(block_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        payload = dict(zip(block_cols, keys, strict=False))
        payload.update(frame_metrics(g))
        rows.append(payload)
    return pd.DataFrame(rows)
