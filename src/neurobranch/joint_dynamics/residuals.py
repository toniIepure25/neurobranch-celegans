"""Causal one-step innovation extraction and residual-structure diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neurobranch.probabilistic.residual import ResidualSeries


@dataclass(frozen=True)
class InnovationSample:
    values: np.ndarray
    targets: np.ndarray
    states: np.ndarray


def lead1_innovations(series: ResidualSeries, *, max_history: int = 1000, t: int | None = None) -> InnovationSample:
    """Return only true one-step prequential residuals.

    The block builder records ``lead == 1`` for the first frame of each fitting
    block, where the parameters were fitted at that same frame.  Later leads
    reuse stale parameters and are deliberately excluded.
    """
    mask = np.isfinite(series.values).all(axis=1) & (series.lead == 1)
    if t is not None:
        mask &= series.targets <= int(t)
    idx = np.flatnonzero(mask)[-int(max_history) :]
    if len(idx) == 0:
        raise ValueError("no lead-1 residuals")
    return InnovationSample(series.values[idx], series.targets[idx], series.states[idx])


def acf_profile(values: np.ndarray, max_lag: int = 20) -> pd.DataFrame:
    arr = np.asarray(values, dtype=np.float64)
    centered = arr - arr.mean(axis=0, keepdims=True)
    rows = []
    for lag in range(1, int(max_lag) + 1):
        vals = []
        for j in range(centered.shape[1]):
            if np.std(centered[:, j]) > 1e-10:
                vals.append(np.corrcoef(centered[:-lag, j], centered[lag:, j])[0, 1])
        rows.append({"lag": lag, "acf_median": float(np.nanmedian(vals)), "acf_abs_median": float(np.nanmedian(np.abs(vals)))})
    return pd.DataFrame(rows)


def temporal_cross_covariance(values: np.ndarray, lag: int = 1) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - arr.mean(axis=0, keepdims=True)
    a, b = arr[:-lag], arr[lag:]
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


def residual_structure(series: ResidualSeries, *, max_history: int = 1000) -> dict[str, float]:
    sample = lead1_innovations(series, max_history=max_history)
    values = sample.values
    centered = values - values.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    cov = (cov + cov.T) / 2
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
    total = float(eig.sum())
    effective_rank = float(total**2 / np.sum(eig**2)) if total > 0 else float("nan")
    top_fraction = float(eig[-1] / total) if total > 0 else float("nan")
    acf = acf_profile(centered, max_lag=20)
    half = len(centered) // 2
    c1 = np.cov(centered[:half], rowvar=False)
    c2 = np.cov(centered[half:], rowvar=False)
    off1 = c1[np.triu_indices_from(c1, k=1)]
    off2 = c2[np.triu_indices_from(c2, k=1)]
    split_corr = float(np.corrcoef(off1, off2)[0, 1]) if np.std(off1) > 0 and np.std(off2) > 0 else float("nan")
    return {
        "n": int(len(values)),
        "trace_cov": float(np.trace(cov)),
        "effective_rank": effective_rank,
        "top_eigenvalue_fraction": top_fraction,
        "acf1": float(acf.loc[acf.lag == 1, "acf_median"].iloc[0]),
        "acf5": float(acf.loc[acf.lag == 5, "acf_median"].iloc[0]),
        "acf10": float(acf.loc[acf.lag == 10, "acf_median"].iloc[0]),
        "acf20": float(acf.loc[acf.lag == 20, "acf_median"].iloc[0]),
        "temporal_cross_cov_lag1": temporal_cross_covariance(centered, 1),
        "covariance_split_half_corr": split_corr,
        "variance_second_half_ratio": float(np.trace(c2) / max(np.trace(c1), 1e-12)),
    }


def conditional_residual_r2(values: np.ndarray, states: np.ndarray, *, ridge: float = 1.0) -> float:
    """Held-out R² for predicting residual mean from the current state."""
    X = np.asarray(states, dtype=np.float64)
    Y = np.asarray(values, dtype=np.float64)
    n = len(X)
    cut = int(n * 0.7)
    if cut < 30 or n - cut < 15:
        return float("nan")
    mx, my = X[:cut].mean(0), Y[:cut].mean(0)
    Xc, Yc = X[:cut] - mx, Y[:cut] - my
    coef = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(X.shape[1]), Xc.T @ Yc)
    pred = (X[cut:] - mx) @ coef + my
    resid = Y[cut:] - pred
    base = Y[cut:] - my
    denom = float(np.sum(base**2))
    return float(1.0 - np.sum(resid**2) / denom) if denom > 0 else float("nan")
