"""Causal innovation residuals and covariance diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

from neurobranch.adaptive.var import FrozenVAR, RollingVAR
from neurobranch.preprocessing import split_starts
from neurobranch.probabilistic.baselines import IndependentARModel, LocalMeanModel


@dataclass
class ResidualSeries:
    model: str
    values: np.ndarray
    targets: np.ndarray
    fit_times: np.ndarray
    lead: np.ndarray
    states: np.ndarray

    def subset(self, t: int, max_history: int = 1000) -> np.ndarray:
        mask = self.targets <= int(t)
        idx = np.flatnonzero(mask)[-int(max_history) :]
        return self.values[idx]

    def history_at(self, t: int, max_history: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        mask = self.targets <= int(t)
        idx = np.flatnonzero(mask)[-int(max_history) :]
        return self.values[idx], self.states[idx]


def _block_residuals(model, X: np.ndarray, *, block: int = 50, start: int) -> tuple[np.ndarray, ...]:
    values, targets, fits, leads, states = [], [], [], [], []
    n = len(X)
    for fit_t in range(int(start), n - 1, int(block)):
        fitted = model.fit_at(X, fit_t)
        if fitted is None:
            continue
        for i in range(fit_t, min(fit_t + block, n - 1)):
            pred = fitted.forecast(X[i], 1)[0]
            values.append(X[i + 1] - pred)
            targets.append(i + 1)
            fits.append(fit_t)
            leads.append(i - fit_t + 1)
            states.append(X[i])
    if not values:
        return (
            np.empty((0, X.shape[1])),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty((0, X.shape[1])),
        )
    return (
        np.asarray(values),
        np.asarray(targets, dtype=np.int64),
        np.asarray(fits, dtype=np.int64),
        np.asarray(leads, dtype=np.int64),
        np.asarray(states),
    )


def adaptive_prequential_residuals(
    data, *, window: int = 500, block: int = 50
) -> ResidualSeries:
    start = max(int(window) + 1, 60)
    values, targets, fits, leads, states = _block_residuals(
        RollingVAR(window=window), data.X, block=block, start=start
    )
    return ResidualSeries("B5_rolling_var_w500", values, targets, fits, leads, states)


def independent_ar_residuals(data, *, window: int, block: int = 50) -> ResidualSeries:
    start = max(int(window) + 1, 60)
    values, targets, fits, leads, states = _block_residuals(
        IndependentARModel(window=window), data.X, block=block, start=start
    )
    return ResidualSeries("B4_independent_ar", values, targets, fits, leads, states)


def frozen_residuals(data, *, ridge: float = 1.0, spectral_cap: float = 0.995) -> ResidualSeries:
    fitted = FrozenVAR.fit(
        data.X,
        split_starts(data, "train", history=1, horizon=1),
        ridge=ridge,
        spectral_cap=spectral_cap,
    )
    n = len(data.X) - 1
    preds = np.stack([fitted.forecast(data.X[i], 1)[0] for i in range(n)])
    targets = np.arange(1, n + 1, dtype=np.int64)
    return ResidualSeries(
        "B1_frozen_var",
        data.X[1 : n + 1] - preds,
        targets,
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.int64),
        data.X[:n],
    )


def persistence_residuals(data) -> ResidualSeries:
    n = len(data.X) - 1
    values = data.X[1 : n + 1] - data.X[:n]
    targets = np.arange(1, n + 1, dtype=np.int64)
    return ResidualSeries(
        "B0_persistence",
        values,
        targets,
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.int64),
        data.X[:n],
    )


def local_mean_residuals(data, *, window: int = 500, ewma_halflife: float | None = None) -> ResidualSeries:
    model = LocalMeanModel(window=window, ewma_halflife=ewma_halflife)
    n = len(data.X) - 1
    preds = []
    for i in range(n):
        fitted = model.fit_at(data.X, i)
        preds.append(
            np.full(data.X.shape[1], np.nan) if fitted is None else fitted.value
        )
    values = data.X[1 : n + 1] - np.asarray(preds, dtype=np.float64)
    targets = np.arange(1, n + 1, dtype=np.int64)
    return ResidualSeries(
        "B2_local_mean",
        values,
        targets,
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.int64),
        data.X[:n],
    )


def center_residuals(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        raise ValueError("no finite residuals")
    mean = arr.mean(axis=0)
    return arr - mean, mean


def regularized_covariance(values: np.ndarray, *, method: str = "ledoit") -> tuple[np.ndarray, dict]:
    """Positive-definite covariance with Ledoit-Wolf or diagonal shrinkage."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) < 3:
        raise ValueError("not enough residuals for covariance")
    centered = arr - arr.mean(axis=0)
    if method == "diagonal":
        cov = np.diag(np.maximum(centered.var(axis=0, ddof=1), 1e-10))
        return cov, {"method": method, "shrinkage": 1.0, "n": int(len(arr))}
    lw = LedoitWolf().fit(centered)
    cov = np.asarray(lw.covariance_, dtype=np.float64)
    shrink = float(lw.shrinkage_)
    eigval, eigvec = np.linalg.eigh((cov + cov.T) / 2)
    floor = max(1e-10, 1e-8 * float(np.max(eigval)))
    eigval = np.clip(eigval, floor, None)
    cov = (eigvec * eigval) @ eigvec.T
    return cov, {
        "method": "ledoit",
        "shrinkage": shrink,
        "n": int(len(arr)),
        "min_eigenvalue": float(eigval.min()),
        "max_eigenvalue": float(eigval.max()),
    }


def residual_diagnostics(series: ResidualSeries) -> dict:
    raw = np.asarray(series.values, dtype=np.float64)
    keep = np.isfinite(raw).all(axis=1)
    values = raw[keep]
    if len(values) < 10:
        return {"n": int(len(values))}
    states = series.states[keep]
    leads = series.lead[keep]
    centered = values - values.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eig = np.clip(np.linalg.eigvalsh((cov + cov.T) / 2), 0, None)
    total = eig.sum()
    pr = float(total**2 / np.sum(eig**2)) if total > 0 else float("nan")
    cond = float(eig.max() / max(eig[eig > 1e-12].min(), 1e-12)) if np.any(eig > 1e-12) else float("nan")
    corr = np.corrcoef(centered, rowvar=False)
    off = corr[np.triu_indices_from(corr, k=1)]
    acf = []
    for lag in (1, 2, 5):
        vals = []
        for j in range(values.shape[1]):
            if np.std(centered[:, j]) > 1e-10:
                vals.append(np.corrcoef(centered[:-lag, j], centered[lag:, j])[0, 1])
        acf.append(float(np.nanmedian(np.abs(vals))) if vals else float("nan"))
    kurt = stats.kurtosis(centered, axis=0, fisher=True, bias=False)
    skew = stats.skew(centered, axis=0, bias=False)
    state_norm = np.linalg.norm(centered, axis=1)
    recent_state = np.linalg.norm(states, axis=1)
    state_corr = (
        float(np.corrcoef(state_norm, recent_state)[0, 1])
        if np.std(state_norm) > 1e-10 and np.std(recent_state) > 1e-10
        else float("nan")
    )
    lead_corr = (
        float(np.corrcoef(np.abs(leads), state_norm)[0, 1])
        if np.std(series.lead) > 0
        else float("nan")
    )
    thirds = np.array_split(centered, 3)
    covs = [np.cov(part, rowvar=False) for part in thirds if len(part) > 3]
    drift = (
        float(np.linalg.norm(covs[-1] - covs[0]) / max(np.linalg.norm(covs[0]), 1e-12))
        if len(covs) == 3
        else float("nan")
    )
    pred_r2 = _predictability_r2(values, states)
    return {
        "n": int(len(values)),
        "mean_norm": float(np.linalg.norm(values.mean(axis=0))),
        "median_neuron_var": float(np.median(values.var(axis=0, ddof=1))),
        "mean_abs_cross_corr": float(np.nanmean(np.abs(off))),
        "max_abs_cross_corr": float(np.nanmax(np.abs(off))),
        "acf_abs_lag1": acf[0],
        "acf_abs_lag2": acf[1],
        "acf_abs_lag5": acf[2],
        "mean_kurtosis": float(np.nanmean(kurt)),
        "mean_abs_skew": float(np.nanmean(np.abs(skew))),
        "state_norm_correlation": state_corr,
        "lead_norm_correlation": lead_corr,
        "covariance_drift_relative": drift,
        "effective_rank": pr,
        "condition_number": cond,
        "predictability_r2": pred_r2,
    }


def _predictability_r2(values: np.ndarray, states: np.ndarray) -> float:
    """Out-of-block R^2 of a ridge map from recent state to innovation."""
    n = min(len(values), len(states))
    if n < 60:
        return float("nan")
    split = int(0.6 * n)
    x_train = states[:split]
    y_train = values[:split]
    x_test = states[split:n]
    y_test = values[split:n]
    mx = x_train.mean(axis=0)
    my = y_train.mean(axis=0)
    xc = x_train - mx
    sd = xc.std(axis=0)
    sd[sd < 1e-8] = 1.0
    xs = xc / sd
    coef = np.linalg.solve(xs.T @ xs + 1.0 * np.eye(xs.shape[1]), xs.T @ (y_train - my))
    pred = ((x_test - mx) / sd) @ coef + my
    sse = float(np.sum((y_test - pred) ** 2))
    sst = float(np.sum((y_test - my) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def covariance_sensitivity(values: np.ndarray) -> pd.DataFrame:
    """Compare covariance estimators and shrinkage levels on a residual pool."""
    centered, _ = center_residuals(values)
    rows = []
    for method in ("ledoit", "diagonal"):
        cov, meta = regularized_covariance(centered, method=method)
        eig = np.linalg.eigvalsh(cov)
        rows.append(
            {
                "method": method,
                "shrinkage": meta["shrinkage"],
                "trace": float(np.trace(cov)),
                "min_eigenvalue": float(eig.min()),
                "max_eigenvalue": float(eig.max()),
                "condition_number": float(eig.max() / max(eig[eig > 1e-12].min(), 1e-12)),
                "offdiag_fraction": float(
                    np.linalg.norm(cov - np.diag(np.diag(cov))) / max(np.linalg.norm(cov), 1e-12)
                ),
            }
        )
    for alpha in (0.1, 0.3, 0.5, 0.8):
        base = np.cov(centered, rowvar=False)
        cov = (1 - alpha) * base + alpha * np.diag(np.diag(base))
        eig = np.linalg.eigvalsh(cov)
        rows.append(
            {
                "method": f"fixed_shrink_{alpha}",
                "shrinkage": alpha,
                "trace": float(np.trace(cov)),
                "min_eigenvalue": float(eig.min()),
                "max_eigenvalue": float(eig.max()),
                "condition_number": float(eig.max() / max(eig[eig > 1e-12].min(), 1e-12)),
                "offdiag_fraction": float(
                    np.linalg.norm(cov - np.diag(np.diag(cov))) / max(np.linalg.norm(cov), 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows)
