"""Metrics and cluster-bootstrap utilities for Phase 6.

Pair-level scoring is primary: repeated stimulations of the same target are
measurements of one latent pair effect, not independent scientific units.
Event-level scoring is retained as a secondary diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats


def _clean(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def pearson(x: ArrayLike, y: ArrayLike) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
        return float("nan")
    return float(stats.pearsonr(a[ok], b[ok]).statistic)


def sign_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.sign(a[ok]) == np.sign(b[ok])))


def pair_table(
    frame: pd.DataFrame,
    *,
    observed: str = "y_true",
    predicted: str = "y_pred",
    min_events: int = 2,
) -> pd.DataFrame:
    """Aggregate event predictions to target-responder pair means."""
    needed = {"target", "neuron", observed, predicted}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"pair_table missing columns {sorted(missing)}")
    grouped = (
        frame.groupby(["target", "neuron"], dropna=False)
        .agg(
            y_true=(observed, "mean"),
            y_pred=(predicted, "mean"),
            n_events=(observed, "size"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n_events"] >= min_events].reset_index(drop=True)
    if "dataset_id" in frame.columns and not grouped.empty:
        # A pair can be measured in several animals.  Assign the modal animal as
        # its bootstrap cluster; the pair mean remains the unit of analysis.
        modes = (
            frame.groupby(["target", "neuron"])["dataset_id"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
            .rename("dataset_id")
            .reset_index()
        )
        grouped = grouped.merge(modes, on=["target", "neuron"], how="left")
    return grouped


def pair_spearman_from_pairs(pairs: pd.DataFrame) -> float:
    return spearman(pairs["y_true"], pairs["y_pred"])


def bootstrap_pair_spearman(
    frame: pd.DataFrame,
    *,
    n_boot: int = 500,
    seed: int = 17,
    min_events: int = 2,
) -> dict[str, float]:
    """Fast animal-cluster bootstrap on pair means.

    The pair table is built once; each bootstrap resamples pair rows by their
    modal animal.  This is the same estimand as resampling events by animal, but
    avoids re-grouping 200k rows on every replicate.
    """
    pairs = pair_table(frame, min_events=min_events)
    if pairs.empty or "dataset_id" not in pairs.columns:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan"), "n_boot": 0.0}
    clusters = pairs["dataset_id"].dropna().unique()
    if len(clusters) < 2:
        return {
            "estimate": pair_spearman_from_pairs(pairs),
            "low": float("nan"),
            "high": float("nan"),
            "n_boot": 0.0,
        }
    rng = np.random.default_rng(seed)
    groups = {c: g for c, g in pairs.groupby("dataset_id")}
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([groups[c] for c in chosen], ignore_index=True)
        values.append(pair_spearman_from_pairs(sample))
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "estimate": pair_spearman_from_pairs(pairs),
        "low": float(np.percentile(arr, 2.5)) if arr.size else float("nan"),
        "high": float(np.percentile(arr, 97.5)) if arr.size else float("nan"),
        "n_boot": float(arr.size),
    }


def paired_pair_bootstrap_delta(
    frame: pd.DataFrame,
    *,
    pred_a: str = "y_pred",
    pred_b: str = "y_pred_baseline",
    n_boot: int = 500,
    seed: int = 23,
    min_events: int = 2,
) -> dict[str, float]:
    """Bootstrap the pair-level Spearman difference by modal animal."""
    a_frame = frame.copy()
    a_frame["y_pred"] = a_frame[pred_a]
    b_frame = frame.copy()
    b_frame["y_pred"] = b_frame[pred_b]
    a = pair_table(a_frame, min_events=min_events)[["target", "neuron", "y_true", "y_pred", "dataset_id"]]
    b = pair_table(b_frame, min_events=min_events)[["target", "neuron", "y_pred"]].rename(
        columns={"y_pred": "y_pred_baseline"}
    )
    merged = a.merge(b, on=["target", "neuron"], how="inner")
    if merged.empty:
        return {"delta": float("nan"), "low": float("nan"), "high": float("nan"), "n_boot": 0.0}

    def delta_metric(df: pd.DataFrame) -> float:
        return spearman(df["y_true"], df["y_pred"]) - spearman(df["y_true"], df["y_pred_baseline"])

    clusters = merged["dataset_id"].dropna().unique()
    if len(clusters) < 2:
        return {
            "delta": float(delta_metric(merged)),
            "low": float("nan"),
            "high": float("nan"),
            "n_boot": 0.0,
        }
    rng = np.random.default_rng(seed)
    groups = {c: g for c, g in merged.groupby("dataset_id")}
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([groups[c] for c in chosen], ignore_index=True)
        values.append(delta_metric(sample))
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "delta": float(delta_metric(merged)),
        "low": float(np.percentile(arr, 2.5)) if arr.size else float("nan"),
        "high": float(np.percentile(arr, 97.5)) if arr.size else float("nan"),
        "n_boot": float(arr.size),
    }


def pair_metrics(frame: pd.DataFrame, *, min_events: int = 2) -> dict[str, float]:
    pairs = pair_table(frame, min_events=min_events)
    if pairs.empty:
        return {
            "n_pairs": 0.0,
            "n_events": float(len(frame)),
            "spearman": float("nan"),
            "pearson": float("nan"),
            "mae": float("nan"),
            "sign_accuracy": float("nan"),
        }
    return {
        "n_pairs": float(len(pairs)),
        "n_events": float(len(frame)),
        "spearman": spearman(pairs["y_true"], pairs["y_pred"]),
        "pearson": pearson(pairs["y_true"], pairs["y_pred"]),
        "mae": float(np.mean(np.abs(pairs["y_true"] - pairs["y_pred"]))),
        "sign_accuracy": sign_accuracy(pairs["y_true"], pairs["y_pred"]),
    }


def event_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = np.asarray(frame["y_true"], dtype=float)
    p = np.asarray(frame["y_pred"], dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return {
            "n": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "spearman": float("nan"),
            "sign_accuracy": float("nan"),
        }
    err = p[ok] - y[ok]
    return {
        "n": float(ok.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "spearman": spearman(y[ok], p[ok]),
        "sign_accuracy": sign_accuracy(y[ok], p[ok]),
    }


def coverage(frame: pd.DataFrame, *, lower: str = "lower", upper: str = "upper") -> float:
    y = np.asarray(frame["y_true"], dtype=float)
    lo = np.asarray(frame[lower], dtype=float)
    hi = np.asarray(frame[upper], dtype=float)
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return float("nan")
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))


def interval_score(
    frame: pd.DataFrame,
    *,
    lower: str = "lower",
    upper: str = "upper",
    alpha: float = 0.10,
) -> float:
    """Winkler interval score (lower is better) for central ``1-alpha`` intervals."""
    y = np.asarray(frame["y_true"], dtype=float)
    lo = np.asarray(frame[lower], dtype=float)
    hi = np.asarray(frame[upper], dtype=float)
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return float("nan")
    y, lo, hi = y[ok], lo[ok], hi[ok]
    width = hi - lo
    penalty_low = (2.0 / alpha) * np.maximum(lo - y, 0.0)
    penalty_high = (2.0 / alpha) * np.maximum(y - hi, 0.0)
    return float(np.mean(width + penalty_low + penalty_high))


def bootstrap_cluster(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    cluster_col: str = "dataset_id",
    n_boot: int = 1000,
    seed: int = 17,
    min_clusters: int = 2,
) -> dict[str, float]:
    """Cluster bootstrap over animals (or another grouping column)."""
    if cluster_col not in frame.columns:
        raise ValueError(f"missing cluster column {cluster_col!r}")
    clusters = frame[cluster_col].dropna().unique()
    if len(clusters) < min_clusters:
        return {"estimate": float(metric(frame)), "low": float("nan"), "high": float("nan"), "n_boot": 0.0}
    rng = np.random.default_rng(seed)
    values = []
    groups = {c: g for c, g in frame.groupby(cluster_col)}
    for _ in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([groups[c] for c in chosen], ignore_index=True)
        values.append(float(metric(sample)))
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"estimate": float(metric(frame)), "low": float("nan"), "high": float("nan"), "n_boot": 0.0}
    return {
        "estimate": float(metric(frame)),
        "low": float(np.percentile(arr, 2.5)),
        "high": float(np.percentile(arr, 97.5)),
        "n_boot": float(arr.size),
    }


def paired_bootstrap_delta(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    pred_a: str = "y_pred",
    pred_b: str = "y_pred_baseline",
    cluster_col: str = "dataset_id",
    n_boot: int = 1000,
    seed: int = 17,
) -> dict[str, float]:
    """Bootstrap ``metric(a) - metric(b)`` with the same animal resample."""
    if cluster_col not in frame.columns:
        raise ValueError(f"missing cluster column {cluster_col!r}")
    clusters = frame[cluster_col].dropna().unique()
    if len(clusters) < 2:
        return {"delta": float("nan"), "low": float("nan"), "high": float("nan"), "n_boot": 0.0}
    rng = np.random.default_rng(seed)
    groups = {c: g for c, g in frame.groupby(cluster_col)}
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([groups[c] for c in chosen], ignore_index=True)
        a = sample.copy()
        b = sample.copy()
        a["y_pred"] = a[pred_a]
        b["y_pred"] = b[pred_b]
        values.append(float(metric(a) - metric(b)))
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"delta": float("nan"), "low": float("nan"), "high": float("nan"), "n_boot": 0.0}
    return {
        "delta": float(np.mean(arr)),
        "low": float(np.percentile(arr, 2.5)),
        "high": float(np.percentile(arr, 97.5)),
        "n_boot": float(arr.size),
    }


def summarize_metric(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    cluster_col: str = "dataset_id",
    n_boot: int = 1000,
    seed: int = 17,
) -> dict[str, Any]:
    point = float(metric(frame))
    ci = bootstrap_cluster(frame, metric, cluster_col=cluster_col, n_boot=n_boot, seed=seed)
    return {
        "value": point,
        "ci_low": ci["low"],
        "ci_high": ci["high"],
        "n_boot": ci["n_boot"],
    }
