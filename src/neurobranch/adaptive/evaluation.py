"""Prequential evaluation utilities for frozen versus adaptive dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.adaptive.var import EWMAVAR, FrozenVAR, RollingVAR
from neurobranch.preprocessing import split_starts


def block_bootstrap_ci(
    values: np.ndarray,
    *,
    block: int = 5,
    n_boot: int = 1000,
    seed: int = 17,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block), len(arr)))
    n_blocks = int(np.ceil(len(arr) / block))
    means = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        starts = rng.integers(0, len(arr) - block + 1, size=n_blocks)
        sample = np.concatenate([arr[s : s + block] for s in starts])[: len(arr)]
        means[b] = sample.mean()
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def evaluation_landmarks(data, horizon_max: int = 100, stride: int = 10) -> np.ndarray:
    valid = split_starts(data, "test", history=2, horizon=horizon_max)
    if len(valid) == 0:
        return valid
    return valid[:: max(1, int(stride))]


def _distribution_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    x0: np.ndarray,
    train_reference: np.ndarray,
) -> dict[str, float]:
    """Distributional fidelity at the final horizon, without test selection."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    finite = np.isfinite(pred).all(axis=1)
    if not finite.all():
        return {
            "finite_fraction": float(finite.mean()),
            "mean_abs_bias": float("nan"),
            "std_ratio": float("nan"),
            "autocorr_error": float("nan"),
            "covariance_relative_error": float("nan"),
            "ood_fraction": float("nan"),
        }
    pred_mean = pred.mean(axis=0)
    true_mean = truth.mean(axis=0)
    pred_std = pred.std(axis=0)
    true_std = truth.std(axis=0)
    good = true_std > 1e-8
    std_ratio = float(np.median(pred_std[good] / true_std[good])) if good.any() else float("nan")
    bias = float(np.mean(np.abs(pred_mean - true_mean)))
    # Lag-one autocorrelation across forecast steps, averaged over neurons.
    pc = []
    tc = []
    for j in range(pred.shape[1]):
        a = pred[:, j]
        b = truth[:, j]
        if np.std(a) > 1e-8:
            pc.append(np.corrcoef(a[:-1], a[1:])[0, 1])
        if np.std(b) > 1e-8:
            tc.append(np.corrcoef(b[:-1], b[1:])[0, 1])
    ac_err = float(abs(np.nanmean(pc) - np.nanmean(tc))) if pc and tc else float("nan")
    if pred.shape[0] < 3:
        cov_err = float("nan")
    else:
        cov_p = np.cov(pred, rowvar=False)
        cov_t = np.cov(truth, rowvar=False)
        denom = float(np.linalg.norm(cov_t))
        cov_err = float(np.linalg.norm(cov_p - cov_t) / denom) if denom > 0 else float("nan")
    q_lo = np.quantile(train_reference, 0.01, axis=0)
    q_hi = np.quantile(train_reference, 0.99, axis=0)
    inside = (pred >= q_lo) & (pred <= q_hi)
    return {
        "finite_fraction": 1.0,
        "mean_abs_bias": bias,
        "std_ratio": std_ratio,
        "autocorr_error": ac_err,
        "covariance_relative_error": cov_err,
        "ood_fraction": float(1.0 - inside.mean()),
    }


@dataclass
class ModelSpec:
    name: str
    factory: Any
    kind: str  # frozen, rolling, ewma


def default_models() -> list[ModelSpec]:
    return [
        ModelSpec("frozen_var", lambda: None, "frozen"),
        ModelSpec("rolling_w200", lambda: RollingVAR(window=200), "rolling"),
        ModelSpec("rolling_w500", lambda: RollingVAR(window=500), "rolling"),
        ModelSpec("rolling_w1000", lambda: RollingVAR(window=1000), "rolling"),
        ModelSpec("ewma_h100", lambda: EWMAVAR(halflife=100), "ewma"),
        ModelSpec("ewma_h300", lambda: EWMAVAR(halflife=300), "ewma"),
        ModelSpec("ewma_h1000", lambda: EWMAVAR(halflife=1000), "ewma"),
    ]


def evaluate_session(
    data,
    *,
    horizon_max: int = 100,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 50, 100),
    stride: int = 10,
    ridge: float = 1.0,
    spectral_cap: float = 0.995,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-landmark, per-neuron (H20) and update rows for one session."""
    landmarks = evaluation_landmarks(data, horizon_max=horizon_max, stride=stride)
    # Existing validated baseline: fit on the train split only, then freeze.
    frozen_starts = split_starts(data, "train", history=1, horizon=1)
    frozen = FrozenVAR.fit(data.X, frozen_starts, ridge=ridge, spectral_cap=spectral_cap)
    models: dict[str, Any] = {"frozen_var": frozen}
    specs = default_models()
    for spec in specs:
        if spec.kind != "frozen":
            models[spec.name] = spec.factory()

    train_ref = data.X[data.masks["train"]]
    rows: list[dict[str, Any]] = []
    neuron_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []

    for t in landmarks:
        x0 = data.X[int(t)].copy()
        truth_all = data.X[int(t) + 1 : int(t) + horizon_max + 1].copy()
        for name, model in models.items():
            started = perf_counter()
            fitted = model if name == "frozen_var" else model.fit_at(data.X, int(t))
            elapsed = perf_counter() - started
            if fitted is None:
                continue
            pred = fitted.forecast(x0, horizon_max)
            radius = float(fitted.spectral_radius)
            update_rows.append(
                {
                    "worm_id": int(data.worm_id),
                    "landmark": int(t),
                    "model": name,
                    "update_seconds": float(elapsed),
                    "spectral_radius": radius,
                    "raw_radius": float(fitted.fit_record.get("raw_radius", radius)),
                    "clip_scale": float(fitted.fit_record.get("scale", 1.0)),
                    "n_transitions": int(fitted.fit_record.get("n_transitions", len(frozen_starts))),
                }
            )
            for h in horizons:
                p = pred[h - 1]
                y = truth_all[h - 1]
                mse = float(np.mean((p - y) ** 2))
                p_mse = float(np.mean((x0 - y) ** 2))
                dist = _distribution_metrics(pred[:h], truth_all[:h], x0, train_ref)
                rows.append(
                    {
                        "worm_id": int(data.worm_id),
                        "landmark": int(t),
                        "model": name,
                        "horizon": int(h),
                        "mse": mse,
                        "persistence_mse": p_mse,
                        "skill_vs_persistence": 1.0 - mse / p_mse if p_mse > 0 else float("nan"),
                        **dist,
                    }
                )
            h = 20
            p = pred[h - 1]
            y = truth_all[h - 1]
            for j, neuron in enumerate(data.predictor_names):
                mse = float((p[j] - y[j]) ** 2)
                p_mse = float((x0[j] - y[j]) ** 2)
                support_lo = np.quantile(train_ref[:, j], 0.01)
                support_hi = np.quantile(train_ref[:, j], 0.99)
                neuron_rows.append(
                    {
                        "worm_id": int(data.worm_id),
                        "landmark": int(t),
                        "model": name,
                        "neuron": neuron,
                        "neuron_index": int(j),
                        "mse_h20": mse,
                        "persistence_mse_h20": p_mse,
                        "skill_vs_persistence_h20": 1.0 - mse / p_mse if p_mse > 0 else float("nan"),
                        "ood_h20": float(not (support_lo <= p[j] <= support_hi)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(neuron_rows), pd.DataFrame(update_rows)


def paired_summary(per_landmark: pd.DataFrame, reference: str = "frozen_var") -> pd.DataFrame:
    """Paired MSE improvement of each model versus the frozen reference at H20."""
    h20 = per_landmark[per_landmark.horizon == 20]
    pivot = h20.pivot_table(
        index=["worm_id", "landmark"], columns="model", values="mse", aggfunc="mean"
    )
    if reference not in pivot:
        return pd.DataFrame()
    rows = []
    for model in pivot.columns:
        if model == reference:
            continue
        diff = (pivot[reference] - pivot[model]).dropna()
        rel = (1.0 - pivot[model] / pivot[reference]).dropna()
        mean, lo, hi = block_bootstrap_ci(diff.to_numpy(), block=5, n_boot=1000)
        rel_mean, rel_lo, rel_hi = block_bootstrap_ci(rel.to_numpy(), block=5, n_boot=1000)
        rows.append(
            {
                "model": model,
                "reference": reference,
                "n_pairs": int(len(diff)),
                "mean_mse_improvement": mean,
                "ci_low": lo,
                "ci_high": hi,
                "mean_relative_improvement": rel_mean,
                "relative_ci_low": rel_lo,
                "relative_ci_high": rel_hi,
                "fraction_landmarks_better": float((diff > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_mse_improvement", ascending=False)
