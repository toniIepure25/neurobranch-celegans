"""Mechanism analyses for SAAV and the compact head-steering circuit.

These are observational encoding/decoding analyses.  They can show incremental
predictive information and cross-animal consistency, not causal necessity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from neurobranch.behavior.protocols import event_metrics, paired_cluster_bootstrap, prior_shift
from neurobranch.behavior.saav_alignment import AlignedDataset


@dataclass
class MechanismResult:
    rows: list[dict[str, Any]]
    predictions: pd.DataFrame
    stability: pd.DataFrame
    animal_effects: pd.DataFrame


def window_summary(x: np.ndarray) -> np.ndarray:
    """Six descriptive statistics per channel for a set of windows."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError("x must be events × frames × channels")
    n, t, c = x.shape
    tt = np.arange(t, dtype=float)
    tt = (tt - tt.mean()) / max(float(np.std(tt)), 1e-9)
    out = np.empty((n, 6 * c), dtype=float)
    for k in range(c):
        v = x[:, :, k]
        out[:, 6 * k + 0] = np.nanmean(v, axis=1)
        out[:, 6 * k + 1] = np.nanstd(v, axis=1)
        out[:, 6 * k + 2] = v[:, -1]
        out[:, 6 * k + 3] = np.nanmin(v, axis=1)
        out[:, 6 * k + 4] = np.nanmax(v, axis=1)
        out[:, 6 * k + 5] = np.nanmean(v * tt[None, :], axis=1)
    return out


def _animal_fold_assignment(animals: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    frame = pd.DataFrame({"animal_id": animals, "y": y})
    counts = frame.groupby("animal_id").agg(n=("y", "size"), dorsal=("y", "sum")).reset_index()
    counts = counts.sort_values(["dorsal", "n"], ascending=False, kind="mergesort")
    dorsal = np.zeros(n_splits)
    total = np.zeros(n_splits)
    assignment: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        k = int(np.lexsort((total, dorsal))[0])
        assignment[row.animal_id] = k
        dorsal[k] += row.dorsal
        total[k] += row.n
    return np.asarray([assignment[a] for a in animals], dtype=int)


def crossfit_predictions(
    x: np.ndarray,
    y: np.ndarray,
    animals: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 17,
    C: float = 0.05,
) -> np.ndarray:
    """Animal-disjoint OOF predictions with train-only scaling and prior shift."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    animals = np.asarray(animals).astype(str)
    folds = _animal_fold_assignment(animals, y, n_splits, seed)
    p = np.full(len(y), np.nan)
    for k in range(n_splits):
        test = np.flatnonzero(folds == k)
        train_pool = np.flatnonzero(folds != k)
        if len(test) == 0 or len(train_pool) == 0 or len(np.unique(y[train_pool])) < 2:
            continue
        scaler = StandardScaler().fit(x[train_pool])
        model = LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=seed)
        model.fit(scaler.transform(x[train_pool]), y[train_pool])
        p_test = model.predict_proba(scaler.transform(x[test]))[:, 1]
        p[test] = prior_shift(p_test, float(np.mean(y[train_pool])))
    return p


def _bootstrap_p(delta: np.ndarray, *, seed: int = 17, n_boot: int = 2000) -> float:
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = np.mean(rng.choice(delta, size=len(delta), replace=True))
    return float(2 * min(np.mean(vals <= 0), np.mean(vals >= 0)))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni family-wise adjustment."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def stability_by_channel(
    data: AlignedDataset,
    channels: tuple[str, ...],
    *,
    seed: int = 17,
    n_boot: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-animal dorsal-minus-ventral activity effects and their stability."""
    animal_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for channel in channels:
        values = data.channel(channel)
        for animal in sorted(data.meta["animal_id"].unique()):
            mask = data.meta["animal_id"].to_numpy() == animal
            finite = np.isfinite(values[mask]).all(axis=1)
            idx = np.flatnonzero(mask)[finite]
            y = data.y[idx]
            if len(idx) < 3 or len(np.unique(y)) < 2:
                continue
            activity = np.nanmean(values[idx], axis=1)
            a = activity[y == 1]
            v = activity[y == 0]
            if len(a) < 1 or len(v) < 1:
                continue
            diff = float(np.mean(a) - np.mean(v))
            pooled_sd = float(np.sqrt((np.var(a, ddof=1) + np.var(v, ddof=1)) / 2)) if len(a) > 1 and len(v) > 1 else float("nan")
            g = diff / pooled_sd if np.isfinite(pooled_sd) and pooled_sd > 0 else float("nan")
            animal_rows.append(
                {
                    "channel": channel,
                    "animal_id": animal,
                    "n_dorsal": int(len(a)),
                    "n_ventral": int(len(v)),
                    "mean_dorsal": float(np.mean(a)),
                    "mean_ventral": float(np.mean(v)),
                    "difference": diff,
                    "hedges_g": g,
                }
            )
        animal_frame = pd.DataFrame(animal_rows)
        if animal_frame.empty or "channel" not in animal_frame.columns:
            continue
        animal_frame = animal_frame[animal_frame["channel"] == channel]
        if animal_frame.empty:
            continue
        effects = animal_frame["difference"].to_numpy()
        rng = np.random.default_rng(seed)
        boot = np.array([np.mean(rng.choice(effects, size=len(effects), replace=True)) for _ in range(n_boot)])
        stability_rows.append(
            {
                "channel": channel,
                "n_animals": int(len(effects)),
                "n_dorsal_events": int(animal_frame["n_dorsal"].sum()),
                "n_ventral_events": int(animal_frame["n_ventral"].sum()),
                "mean_dorsal_minus_ventral": float(np.mean(effects)),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "fraction_positive": float(np.mean(effects > 0)),
                "median_hedges_g": float(np.nanmedian(animal_frame["hedges_g"])),
            }
        )
    return pd.DataFrame(stability_rows), pd.DataFrame(animal_rows)


def incremental_channel_tests(
    data: AlignedDataset,
    channels: tuple[str, ...],
    *,
    behavior_channels: tuple[str, ...] = ("hc",),
    roll_frames: int = 0,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test each channel's incremental information over head curvature."""
    behavior_x = np.stack([data.channel(c) for c in behavior_channels], axis=-1)
    behavior = window_summary(behavior_x)
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    animals = data.meta["animal_id"].astype(str).to_numpy()
    for channel in channels:
        raw = data.channel(channel)
        if roll_frames:
            raw = np.roll(raw, shift=roll_frames, axis=1)
        finite = np.isfinite(raw).all(axis=1)
        if finite.sum() < 20:
            continue
        x = np.column_stack([behavior[finite], window_summary(raw[finite, :, None])])
        y = data.y[finite]
        a = animals[finite]
        if len(np.unique(y)) < 2 or len(np.unique(a)) < 3:
            continue
        p0 = crossfit_predictions(behavior[finite], y, a, seed=seed)
        p1 = crossfit_predictions(x, y, a, seed=seed)
        common = np.isfinite(p0) & np.isfinite(p1)
        yc, a_c = y[common], a[common]
        p0c, p1c = p0[common], p1[common]
        m0 = event_metrics(yc, p0c)
        m1 = event_metrics(yc, p1c)
        for metric in ("log_loss", "brier", "auroc", "balanced_accuracy"):
            res = paired_cluster_bootstrap(yc, p0c, p1c, a_c, metric=metric, n_boot=1000, seed=seed)
            rows.append(
                {
                    "channel": channel,
                    "control_roll_frames": roll_frames,
                    "metric": metric,
                    "behavior_only": m0.get(metric),
                    "with_channel": m1.get(metric),
                    "delta": res["delta"],
                    "ci_low": res["ci_low"],
                    "ci_high": res["ci_high"],
                    "n_events": int(common.sum()),
                    "n_animals": int(len(np.unique(a_c))),
                    "bootstrap_p": res.get("p_two_sided", float("nan")),
                }
            )
        p1_full = np.full(len(data.y), np.nan)
        p0_full = np.full(len(data.y), np.nan)
        p1_full[finite] = p1
        p0_full[finite] = p0
        predictions[channel] = p1_full
        predictions[f"{channel}__behavior"] = p0_full
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def clustered_bootstrap_p(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    metric: str,
    *,
    seed: int = 17,
    n_boot: int = 2000,
) -> float:
    """Two-sided animal-cluster bootstrap p-value for a paired metric delta."""
    from neurobranch.behavior.protocols import event_metrics

    y = np.asarray(y, dtype=int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    groups = np.asarray(groups)
    observed = event_metrics(y, p_b).get(metric, np.nan) - event_metrics(y, p_a).get(metric, np.nan)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    deltas: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(
            event_metrics(y[idx], p_b[idx]).get(metric, np.nan)
            - event_metrics(y[idx], p_a[idx]).get(metric, np.nan)
        )
    if not deltas or not np.isfinite(observed):
        return float("nan")
    arr = np.asarray(deltas)
    return float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0)))
