"""Prospective reversal prediction with strictly causal features.

The primary task is animal-disjoint: train on all but one animal and predict
whether a reversal begins in ``(t, t + H]`` seconds.  Every feature at ``t`` uses
only frames ``<= t``.  Normalisation, PCA and regularisation are fit inside each
training fold.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from neurobranch.data.baskoylu import FreelyMovingRecording

HORIZONS_S: tuple[float, ...] = (3.0, 6.0, 12.0)
PRIMARY_HORIZON_S = 6.0
LAGS = (0, 1, 2, 4, 8)
WINDOWS = (5, 10, 20)
ROLLING = (5, 10, 20)
MIN_HISTORY_FRAMES = 10

BEHAVIOR_MODES = (
    "constant",
    "state",
    "behavior",
    "behavior_no_counts",
    "behavior_no_rolling",
    "behavior_quadratic",
    "markov",
)
NEURAL_MODES = (
    "neural",
    "behavior_neural",
    "behavior_neural_pca",
    "behavior_neural_shift",
    "behavior_neural_no_counts",
    "behavior_neural_no_rolling",
    "behavior_neural_quadratic",
)
IDENTIFIED_MODES = ("behavior_identified", "behavior_all")


def horizon_frames(rec: FreelyMovingRecording, seconds: float) -> int:
    return int(np.ceil(seconds / rec.dt - 1e-9))


def _onset_vector(rec: FreelyMovingRecording) -> np.ndarray:
    v = np.zeros(rec.n_frames, dtype=np.int8)
    v[np.clip(rec.reversal_onsets, 0, rec.n_frames - 1)] = 1
    return v


def eligible_frames(rec: FreelyMovingRecording, h: int) -> np.ndarray:
    """Non-reversing frames with a complete future window and some past."""
    ok = np.zeros(rec.n_frames, dtype=bool)
    ok[MIN_HISTORY_FRAMES : rec.n_frames - h] = True
    ok &= rec.reversal_vec == 0
    return np.flatnonzero(ok)


def horizon_labels(rec: FreelyMovingRecording, idx: np.ndarray, h: int) -> np.ndarray:
    onset = _onset_vector(rec)
    # Any onset in t+1 ... t+h.  The current frame is excluded by construction.
    csum = np.r_[0, np.cumsum(onset)]
    return (csum[idx + h + 1] - csum[idx + 1] > 0).astype(np.int8)


def causal_zscore(x: np.ndarray) -> np.ndarray:
    """Expanding z-score using current and past samples only."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    c1 = np.cumsum(x, axis=0)
    c2 = np.cumsum(x * x, axis=0)
    count = np.arange(1, n + 1, dtype=np.float64)[:, None]
    mean = c1 / count
    var = np.maximum(c2 / count - mean * mean, 0.0)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0
    z = (x - mean) / std
    z[0] = 0.0
    return z


def _roll(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean and SD for every frame."""
    x = np.asarray(x, dtype=np.float64).ravel()
    c1 = np.r_[0.0, np.cumsum(x)]
    c2 = np.r_[0.0, np.cumsum(x * x)]
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - window)
    n = (idx - lo).astype(float)
    mean = (c1[idx] - c1[lo]) / n
    var = np.maximum((c2[idx] - c2[lo]) / n - mean * mean, 0.0)
    return mean, np.sqrt(var)


def _lag(x: np.ndarray, lag: int, idx: np.ndarray) -> np.ndarray:
    src = np.clip(idx - lag, 0, len(x) - 1)
    return np.asarray(x)[src]


def _causal_shift(x: np.ndarray, frames: int = 211) -> np.ndarray:
    """Delay features by ``frames`` without wrapping future rows to the start."""
    out = np.empty_like(x)
    out[:frames] = x[0]
    out[frames:] = x[:-frames]
    return out


def _run_age(rec: FreelyMovingRecording) -> np.ndarray:
    """Frames since the last reversal offset (0 while reversing)."""
    age = np.zeros(rec.n_frames, dtype=np.float64)
    last_end = 0
    ends = set(int(x) for x in rec.reversal_offsets)
    for t in range(rec.n_frames):
        if t in ends:
            last_end = t
        age[t] = 0.0 if rec.reversal_vec[t] else float(t - last_end)
    return age


def behavior_matrix(rec: FreelyMovingRecording, *, state_only: bool = False) -> tuple[np.ndarray, list[str]]:
    """Current state plus causal history features.  No future frames."""
    v = rec.velocity.astype(float)
    speed = np.abs(v)
    accel = np.r_[0.0, np.diff(v)]
    ang = rec.angular_velocity.astype(float)
    head = rec.head_angle.astype(float)
    dist = rec.distance_from_center.astype(float)
    age_frames = _run_age(rec)
    age_s = age_frames * rec.dt
    onset = _onset_vector(rec).astype(float)
    onset_count = np.r_[0.0, np.cumsum(onset)]

    cols: list[np.ndarray] = [
        v,
        speed,
        accel,
        np.sin(head),
        np.cos(head),
        ang,
        np.abs(ang),
        dist,
        age_s,
        np.log1p(age_frames),
    ]
    names = [
        "velocity",
        "speed",
        "accel",
        "head_sin",
        "head_cos",
        "angular_velocity",
        "abs_angular_velocity",
        "distance_from_center",
        "forward_age_s",
        "log_forward_age",
    ]
    if state_only:
        return np.column_stack(cols).astype(np.float64), names

    for lag in (1, 2, 4, 8):
        cols.append(_lag(v, lag, np.arange(rec.n_frames)))
        names.append(f"velocity_lag{lag}")
    for w in WINDOWS:
        m, s = _roll(v, w)
        cols.extend([m, s])
        names.extend([f"velocity_mean_{w}", f"velocity_sd_{w}"])
        m, s = _roll(ang, w)
        cols.extend([m, s])
        names.extend([f"angular_mean_{w}", f"angular_sd_{w}"])
        m, s = _roll(dist, w)
        cols.extend([m, s])
        names.extend([f"distance_mean_{w}", f"distance_sd_{w}"])
        cols.append(_lag(v, w, np.arange(rec.n_frames)) - v)
        names.append(f"velocity_delta_{w}")
        c = onset_count[np.arange(1, rec.n_frames + 1)] - onset_count[np.maximum(0, np.arange(rec.n_frames) - w + 1)]
        cols.append(c)
        names.append(f"reversals_in_{w}")
    return np.column_stack(cols).astype(np.float64), names


def population_neural_matrix(rec: FreelyMovingRecording) -> tuple[np.ndarray, list[str]]:
    """Permutation-invariant summaries of causally normalised neural activity.

    ROI indices are not identities across animals.  These summaries (mean, SD,
    quantiles, extremes, active fractions and their temporal differences) are a
    deliberately conservative population-level representation.
    """
    z = causal_zscore(rec.traces_raw)
    n = np.arange(rec.n_frames)
    cols: list[np.ndarray] = []
    names: list[str] = []
    for lag in LAGS:
        q = z[np.clip(n - lag, 0, rec.n_frames - 1)]
        stats = [
            np.mean(q, axis=1),
            np.std(q, axis=1),
            np.quantile(q, 0.10, axis=1),
            np.quantile(q, 0.25, axis=1),
            np.quantile(q, 0.50, axis=1),
            np.quantile(q, 0.75, axis=1),
            np.quantile(q, 0.90, axis=1),
            np.max(q, axis=1),
            np.min(q, axis=1),
            np.mean(q > 1.0, axis=1),
            np.mean(q < -1.0, axis=1),
        ]
        cols.extend(stats)
        prefix = f"neural_pop_lag{lag}"
        names.extend(
            [
                f"{prefix}_mean",
                f"{prefix}_sd",
                f"{prefix}_q10",
                f"{prefix}_q25",
                f"{prefix}_q50",
                f"{prefix}_q75",
                f"{prefix}_q90",
                f"{prefix}_max",
                f"{prefix}_min",
                f"{prefix}_frac_gt1",
                f"{prefix}_frac_lt-1",
            ]
        )
        if lag:
            cols.append(stats[0] - np.mean(z, axis=1))
            names.append(f"neural_pop_mean_delta_lag{lag}")
    return np.column_stack(cols).astype(np.float64), names


def shared_identity_labels(recs: Iterable[FreelyMovingRecording], min_fraction: float = 0.8) -> list[str]:
    counts: dict[str, int] = {}
    recs = list(recs)
    for rec in recs:
        for name in set(rec.identified_indices):
            counts[name] = counts.get(name, 0) + 1
    need = int(np.ceil(min_fraction * len(recs)))
    return sorted(n for n, c in counts.items() if c >= need)


def identified_neural_matrix(
    rec: FreelyMovingRecording, labels: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Causally normalised identified-neuron traces; missing identities are 0.

    Zero is the training mean after causal z-scoring.  No principal component or
    missing-value statistic is fit on held-out animals here.
    """
    t = rec.n_frames
    raw = np.full((t, len(labels)), np.nan, dtype=np.float64)
    idx = rec.identified_indices
    for j, name in enumerate(labels):
        i = idx.get(name)
        if i is not None:
            raw[:, j] = rec.traces_raw[:, i]
    z = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        good = np.isfinite(col)
        if good.sum() < 2:
            continue
        zz = causal_zscore(np.where(good, col, np.nan_to_num(col)))[:, 0]
        z[:, j] = np.where(good, zz, 0.0)
    names = [f"identified_{name}" for name in labels]
    return z, names


@dataclass
class AnimalDesign:
    animal_id: str
    idx: np.ndarray
    y: np.ndarray
    behavior: np.ndarray
    behavior_no_counts: np.ndarray
    behavior_no_rolling: np.ndarray
    behavior_quadratic: np.ndarray
    state: np.ndarray
    population: np.ndarray
    population_shift: np.ndarray
    identified: np.ndarray | None
    age: np.ndarray


def build_designs(
    recs: list[FreelyMovingRecording], horizon_s: float, *, identity_fraction: float = 0.8
) -> tuple[list[AnimalDesign], list[str]]:
    labels = shared_identity_labels(recs, identity_fraction)
    designs = []
    for rec in recs:
        h = horizon_frames(rec, horizon_s)
        idx = eligible_frames(rec, h)
        y = horizon_labels(rec, idx, h)
        b, bnames = behavior_matrix(rec)
        no_counts = b[:, [i for i, n in enumerate(bnames) if not n.startswith("reversals_in_")]]
        no_rolling = b[
            :,
            [
                i
                for i, n in enumerate(bnames)
                if not any(
                    token in n
                    for token in ("_mean_", "_sd_", "_delta_", "reversals_in_")
                )
            ],
        ]
        s, _ = behavior_matrix(rec, state_only=True)
        quad = PolynomialFeatures(degree=2, include_bias=False).fit_transform(s)
        bq = np.hstack([b, quad])
        p, _ = population_neural_matrix(rec)
        ident = identified_neural_matrix(rec, labels)[0] if labels else None
        age = _run_age(rec)[idx]
        designs.append(
            AnimalDesign(
                animal_id=rec.animal_id,
                idx=idx,
                y=y,
                behavior=b[idx],
                behavior_no_counts=no_counts[idx],
                behavior_no_rolling=no_rolling[idx],
                behavior_quadratic=bq[idx],
                state=s[idx],
                population=p[idx],
                population_shift=_causal_shift(p)[idx],
                identified=ident[idx] if ident is not None else None,
                age=age,
            )
        )
    return designs, labels


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    Xte: np.ndarray,
    *,
    seed: int = 17,
    C: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a prespecified L2 logistic model.

    ``C=0.1`` is fixed before seeing test metrics; no inner search is needed, so
    the expensive repeated fits cannot tune on the held-out animal.  A
    sensitivity run over ``C in {0.03, 0.3}`` is reported separately.
    """
    scaler = StandardScaler().fit(X)
    Xs = np.nan_to_num(scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    Xte_s = np.nan_to_num(scaler.transform(Xte), nan=0.0, posinf=0.0, neginf=0.0)
    if len(np.unique(y)) < 2:
        return np.full(len(Xte), float(np.mean(y))), {"C": None, "constant": True}
    model = LogisticRegression(C=C, max_iter=1000, tol=1e-3, solver="lbfgs", random_state=seed)
    model.fit(Xs, y)
    p = model.predict_proba(Xte_s)[:, 1]
    return p, {"C": C, "constant": False, "n_features": int(X.shape[1])}


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    out = {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, (p >= 0.5).astype(int))),
        "precision": float(precision_score(y, (p >= 0.5).astype(int), zero_division=0)),
        "recall": float(recall_score(y, (p >= 0.5).astype(int), zero_division=0)),
    }
    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
        # Test-set calibration diagnostic only; never used to select a model.
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)
        cal = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs").fit(logit, y)
        out["calibration_slope"] = float(cal.coef_[0, 0])
        out["calibration_intercept"] = float(cal.intercept_[0])
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
        out["calibration_slope"] = float("nan")
        out["calibration_intercept"] = float("nan")
    return out


def _markov_predict(design: AnimalDesign, train_idx: list[int], test_design: AnimalDesign) -> np.ndarray:
    age_train = np.concatenate([np.asarray(d.age) for d in train_idx])
    y_train = np.concatenate([np.asarray(d.y) for d in train_idx])
    edges = np.unique(np.quantile(age_train, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    if len(edges) < 3:
        return np.full(len(test_design.y), float(np.mean(y_train)))
    bins_train = np.clip(np.digitize(age_train, edges[1:-1]), 0, len(edges) - 2)
    bins_test = np.clip(np.digitize(test_design.age, edges[1:-1]), 0, len(edges) - 2)
    p = np.full(len(edges) - 1, float(np.mean(y_train)))
    for b in range(len(p)):
        m = bins_train == b
        if m.sum() >= 20:
            p[b] = (y_train[m].sum() + 1.0) / (m.sum() + 2.0)
    return p[bins_test]


def evaluate_animal_disjoint(
    recs: list[FreelyMovingRecording],
    horizon_s: float,
    *,
    modes: tuple[str, ...] = BEHAVIOR_MODES + NEURAL_MODES + IDENTIFIED_MODES,
    seed: int = 17,
    n_bootstrap: int = 10000,
    c_value: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Leave-one-animal-out evaluation.  Returns metrics, per-animal deltas, summary."""
    designs, labels = build_designs(recs, horizon_s)
    all_modes = [m for m in modes if m not in IDENTIFIED_MODES or labels]
    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for test_i, test in enumerate(designs):
        train = [d for j, d in enumerate(designs) if j != test_i]
        groups = np.concatenate([np.full(len(d.y), d.animal_id) for d in train])
        y_train = np.concatenate([d.y for d in train])
        Xtr = {
            "state": np.vstack([d.state for d in train]),
            "behavior": np.vstack([d.behavior for d in train]),
            "behavior_no_counts": np.vstack([d.behavior_no_counts for d in train]),
            "behavior_no_rolling": np.vstack([d.behavior_no_rolling for d in train]),
            "behavior_quadratic": np.vstack([d.behavior_quadratic for d in train]),
        }
        Xte = {
            "state": test.state,
            "behavior": test.behavior,
            "behavior_no_counts": test.behavior_no_counts,
            "behavior_no_rolling": test.behavior_no_rolling,
            "behavior_quadratic": test.behavior_quadratic,
        }
        if any(m in all_modes for m in NEURAL_MODES):
            Xtr["population"] = np.vstack([d.population for d in train])
            Xtr["population_shift"] = np.vstack([d.population_shift for d in train])
            Xte["population"] = test.population
            Xte["population_shift"] = test.population_shift
        if labels and ("behavior_identified" in all_modes or "behavior_all" in all_modes):
            Xtr["identified"] = np.vstack([d.identified for d in train])
            Xte["identified"] = test.identified
        for mode in all_modes:
            if mode == "constant":
                p = np.full(len(test.y), float(np.mean(y_train)))
                info = {"constant": True}
            elif mode == "markov":
                p = _markov_predict(test, train, test)
                info = {"constant": False}
            else:
                if mode in {
                    "state",
                    "behavior",
                    "behavior_no_counts",
                    "behavior_no_rolling",
                    "behavior_quadratic",
                }:
                    a, b = Xtr[mode], Xte[mode]
                elif mode == "neural":
                    a, b = Xtr["population"], Xte["population"]
                elif mode == "behavior_neural":
                    a = np.hstack([Xtr["behavior"], Xtr["population"]])
                    b = np.hstack([Xte["behavior"], Xte["population"]])
                elif mode == "behavior_neural_shift":
                    a = np.hstack([Xtr["behavior"], Xtr["population_shift"]])
                    b = np.hstack([Xte["behavior"], Xte["population_shift"]])
                elif mode == "behavior_neural_no_counts":
                    a = np.hstack([Xtr["behavior_no_counts"], Xtr["population"]])
                    b = np.hstack([Xte["behavior_no_counts"], Xte["population"]])
                elif mode == "behavior_neural_no_rolling":
                    a = np.hstack([Xtr["behavior_no_rolling"], Xtr["population"]])
                    b = np.hstack([Xte["behavior_no_rolling"], Xte["population"]])
                elif mode == "behavior_neural_quadratic":
                    a = np.hstack([Xtr["behavior_quadratic"], Xtr["population"]])
                    b = np.hstack([Xte["behavior_quadratic"], Xte["population"]])
                elif mode == "behavior_neural_pca":
                    pca = PCA(n_components=min(10, Xtr["population"].shape[1], len(y_train) - 1), random_state=seed)
                    pca.fit(np.nan_to_num(Xtr["population"]))
                    a = np.hstack([Xtr["behavior"], pca.transform(np.nan_to_num(Xtr["population"]))])
                    b = np.hstack([Xte["behavior"], pca.transform(np.nan_to_num(Xte["population"]))])
                    info = {"pca_components": int(pca.n_components_)}
                elif mode == "behavior_identified":
                    a = np.hstack([Xtr["behavior"], Xtr["identified"]])
                    b = np.hstack([Xte["behavior"], Xte["identified"]])
                elif mode == "behavior_all":
                    a = np.hstack([Xtr["behavior"], Xtr["population"], Xtr["identified"]])
                    b = np.hstack([Xte["behavior"], Xte["population"], Xte["identified"]])
                else:
                    raise ValueError(mode)
                p, info = _fit_logistic(a, y_train, groups, b, seed=seed, C=c_value)
            m = _metrics(test.y, p)
            rows.append({"animal_id": test.animal_id, "horizon_s": horizon_s, "mode": mode, **m, **info})
            if mode in {"behavior", "behavior_neural", "behavior_identified"}:
                for k, (yy, pp) in enumerate(zip(test.y, p)):
                    pred_rows.append(
                        {
                            "animal_id": test.animal_id,
                            "frame": int(test.idx[k]),
                            "horizon_s": horizon_s,
                            "mode": mode,
                            "y": int(yy),
                            "p": float(pp),
                        }
                    )
    metrics = pd.DataFrame(rows)
    piv = metrics.pivot(index="animal_id", columns="mode", values="log_loss")
    deltas = pd.DataFrame(index=piv.index)
    for mode in [
        "state",
        "neural",
        "behavior_neural",
        "behavior_neural_pca",
        "behavior_neural_shift",
        "behavior_neural_no_counts",
        "behavior_neural_no_rolling",
        "behavior_neural_quadratic",
        "behavior_identified",
        "behavior_all",
    ]:
        if mode in piv and "behavior" in piv:
            deltas[f"delta_{mode}"] = piv[mode] - piv["behavior"]
    if {"behavior_no_counts", "behavior_neural_no_counts"}.issubset(piv.columns):
        deltas["delta_neural_within_no_counts"] = piv["behavior_neural_no_counts"] - piv["behavior_no_counts"]
    if {"behavior_no_rolling", "behavior_neural_no_rolling"}.issubset(piv.columns):
        deltas["delta_neural_within_no_rolling"] = (
            piv["behavior_neural_no_rolling"] - piv["behavior_no_rolling"]
        )
    if {"behavior_quadratic", "behavior_neural_quadratic"}.issubset(piv.columns):
        deltas["delta_neural_within_quadratic"] = (
            piv["behavior_neural_quadratic"] - piv["behavior_quadratic"]
        )
    deltas = deltas.reset_index()
    summary: dict[str, Any] = {
        "horizon_s": horizon_s,
        "n_animals": int(len(designs)),
        "n_identified_labels": int(len(labels)),
        "modes": all_modes,
        "aggregate": {},
    }
    for mode, g in metrics.groupby("mode"):
        summary["aggregate"][mode] = {
            "mean_log_loss": float(g["log_loss"].mean()),
            "mean_brier": float(g["brier"].mean()),
            "mean_balanced_accuracy": float(g["balanced_accuracy"].mean()),
            "mean_auprc": float(g["auprc"].mean()),
            "mean_prevalence": float(g["prevalence"].mean()),
            "mean_calibration_slope": float(g["calibration_slope"].mean()),
            "mean_calibration_intercept": float(g["calibration_intercept"].mean()),
        }
    rng = np.random.default_rng(seed)
    for col in [c for c in deltas.columns if c.startswith("delta_")]:
        v = deltas[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        boots = rng.choice(v, size=(n_bootstrap, len(v)), replace=True).mean(axis=1)
        summary["aggregate"][col] = {
            "mean": float(v.mean()),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "n_negative": int((v < 0).sum()),
            "n": int(len(v)),
        }
    return metrics, deltas, summary


def chronological_split(n: int, h: int, train_fraction: float = 0.7, val_fraction: float = 0.15):
    n_train = int(np.floor(train_fraction * n))
    n_val = int(np.floor(val_fraction * n))
    train = np.arange(0, n_train)
    val_start = min(n, n_train + h)
    val = np.arange(val_start, min(n, val_start + n_val))
    test_start = min(n, (val[-1] + 1 + h) if len(val) else val_start + h)
    test = np.arange(test_start, n)
    return train, val, test


def evaluate_within_animal(
    recs: list[FreelyMovingRecording], horizon_s: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chronological forward evaluation, reported separately from LOAO."""
    rows = []
    for rec in recs:
        h = horizon_frames(rec, horizon_s)
        idx = eligible_frames(rec, h)
        y = horizon_labels(rec, idx, h)
        b, _ = behavior_matrix(rec)
        p, _ = population_neural_matrix(rec)
        train, val, test = chronological_split(rec.n_frames, h)
        # Restrict to eligible frames in each segment. Bind explicitly so the
        # closure cannot accidentally capture a later loop's arrays.
        def sel(seg, idx=idx, y=y):
            mask = np.isin(idx, seg)
            return idx[mask], y[mask]
        tri, try_ = sel(train)
        vai, vay = sel(val)
        tei, tey = sel(test)
        if len(tri) < 50 or len(tei) < 20 or len(np.unique(try_)) < 2:
            continue
        Xb_tr, Xb_te = b[tri], b[tei]
        Xn_tr, Xn_te = p[tri], p[tei]
        bmodel = _fit_logistic(Xb_tr, try_, np.zeros(len(try_)), Xb_te, seed=17)
        nmodel = _fit_logistic(np.hstack([Xb_tr, Xn_tr]), try_, np.zeros(len(try_)), np.hstack([Xb_te, Xn_te]), seed=17)
        for mode, (yy, pp) in {"behavior": (tey, bmodel[0]), "behavior_neural": (tey, nmodel[0])}.items():
            rows.append({"animal_id": rec.animal_id, "horizon_s": horizon_s, "mode": mode, **_metrics(yy, pp)})
    m = pd.DataFrame(rows)
    summary: dict[str, Any] = {"horizon_s": horizon_s, "aggregate": {}}
    for mode, g in m.groupby("mode"):
        summary["aggregate"][mode] = {
            "mean_log_loss": float(g["log_loss"].mean()),
            "mean_brier": float(g["brier"].mean()),
            "mean_balanced_accuracy": float(g["balanced_accuracy"].mean()),
        }
    if {"behavior", "behavior_neural"}.issubset(set(m["mode"])):
        piv = m.pivot(index="animal_id", columns="mode", values="log_loss")
        d = (piv["behavior_neural"] - piv["behavior"]).dropna()
        summary["delta"] = {
            "mean": float(d.mean()),
            "n_negative": int((d < 0).sum()),
            "n": int(len(d)),
        }
    return m, summary


def write_results(
    recs: list[FreelyMovingRecording],
    *,
    out_dir: str | Path = "reports/phase3",
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_all, deltas_all, summaries = [], [], {}
    for horizon in HORIZONS_S:
        metrics, deltas, summary = evaluate_animal_disjoint(recs, horizon, n_bootstrap=n_bootstrap)
        metrics.to_csv(out / f"prospective_metrics_{horizon:g}s.csv", index=False)
        deltas.to_csv(out / f"prospective_deltas_{horizon:g}s.csv", index=False)
        metrics_all.append(metrics)
        deltas_all.append(deltas.assign(horizon_s=horizon))
        summaries[f"{horizon:g}s"] = summary
    pd.concat(metrics_all, ignore_index=True).to_csv(out / "prospective_metrics_all.csv", index=False)
    pd.concat(deltas_all, ignore_index=True).to_csv(out / "prospective_deltas_all.csv", index=False)
    within_metrics, within_summary = evaluate_within_animal(recs, PRIMARY_HORIZON_S)
    within_metrics.to_csv(out / "prospective_within_animal_metrics.csv", index=False)
    summaries["within_animal_6s"] = within_summary
    (out / "prospective_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries
