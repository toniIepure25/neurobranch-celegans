"""Causal linear models for the Kato common-state control audit.

The independent models here use the same standardized-feature ridge objective as
the population VAR.  This is deliberately different from the historical weak
ridge in ``IndependentARModel``; the historical model is retained only for
reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neurobranch.adaptive.var import fit_ridge_var, spectral_radius
from neurobranch.kato_interactions.models import FrozenLinearModel, diagonal_copy


def _clean_transitions(
    X: np.ndarray, starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(X, dtype=np.float64)
    idx = np.asarray(starts, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx + 1 < len(arr))]
    if len(idx) < 2:
        raise ValueError("fewer than two causal transitions")
    design = arr[idx]
    target = arr[idx + 1]
    finite = np.isfinite(design).all(axis=1) & np.isfinite(target).all(axis=1)
    return design[finite], target[finite], idx[finite]


def _cap_vector(values: np.ndarray, cap: float | None) -> tuple[np.ndarray, float, bool]:
    raw = float(np.max(np.abs(values)))
    if cap is not None and raw > float(cap):
        return values * (float(cap) / max(raw, 1e-12)), raw, True
    return values, raw, False


@dataclass
class MatchedOwnAR:
    a: np.ndarray
    b: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    window: int
    ridge: float
    fit_record: dict[str, Any] = field(default_factory=dict)

    @property
    def n_parameters(self) -> int:
        return int(2 * len(self.a))

    def forecast_mean(self, x0: np.ndarray, horizon: int) -> np.ndarray:
        x = np.asarray(x0, dtype=np.float64).copy()
        out = np.empty((int(horizon), x.size), dtype=np.float64)
        for step in range(int(horizon)):
            x = self.a * x + self.b
            out[step] = x
        return out


def fit_matched_own_ar(
    X: np.ndarray,
    starts: np.ndarray,
    *,
    window: int,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
) -> MatchedOwnAR:
    design, target, idx = _clean_transitions(X, starts)
    mu = design.mean(axis=0)
    sd = design.std(axis=0)
    sd[sd < 1e-8] = 1.0
    xs = (design - mu) / sd
    yc = target - target.mean(axis=0)
    coef = (xs * yc).sum(axis=0) / ((xs**2).sum(axis=0) + float(ridge))
    a = coef / sd
    a, raw, clipped = _cap_vector(a, spectral_cap)
    b = target.mean(axis=0) - a * mu
    return MatchedOwnAR(
        a=a,
        b=b,
        mu=mu,
        sd=sd,
        window=int(window),
        ridge=float(ridge),
        fit_record={
            "model": "matched_own_ar",
            "window": int(window),
            "ridge": float(ridge),
            "n_transitions": int(len(idx)),
            "first_transition": int(idx.min()),
            "last_transition": int(idx.max()),
            "raw_radius": raw,
            "clipped": bool(clipped),
            "spectral_radius": spectral_radius(np.diag(a)),
            "n_parameters": int(2 * len(a)),
        },
    )


@dataclass
class TrendAR:
    a: np.ndarray
    trend: np.ndarray
    b: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    time_mu: float
    time_sd: float
    window: int
    ridge: float
    fit_record: dict[str, Any] = field(default_factory=dict)

    @property
    def n_parameters(self) -> int:
        return int(3 * len(self.a))

    def forecast_mean(self, x0: np.ndarray, t: int, horizon: int) -> np.ndarray:
        x = np.asarray(x0, dtype=np.float64).copy()
        out = np.empty((int(horizon), x.size), dtype=np.float64)
        for step in range(int(horizon)):
            time_value = float(t + step + 1)
            x = (
                self.b
                + self.a * (x - self.mu) / self.sd
                + self.trend * ((time_value - self.time_mu) / self.time_sd)
            )
            out[step] = x
        return out


def fit_trend_ar(
    X: np.ndarray,
    starts: np.ndarray,
    *,
    window: int,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
) -> TrendAR:
    design, target, idx = _clean_transitions(X, starts)
    mu = design.mean(axis=0)
    sd = design.std(axis=0)
    sd[sd < 1e-8] = 1.0
    time_target = (idx + 1).astype(np.float64)
    time_mu = float(time_target.mean())
    time_sd = float(time_target.std())
    if time_sd < 1e-8:
        time_sd = 1.0
    tscaled = (time_target - time_mu) / time_sd
    n = design.shape[1]
    a = np.empty(n, dtype=np.float64)
    trend = np.empty(n, dtype=np.float64)
    b = np.empty(n, dtype=np.float64)
    for j in range(n):
        xs = (design[:, j] - mu[j]) / sd[j]
        design2 = np.column_stack([xs, tscaled])
        y = target[:, j] - target[:, j].mean()
        coef = np.linalg.solve(
            design2.T @ design2 + float(ridge) * np.eye(2),
            design2.T @ y,
        )
        a[j] = coef[0]
        trend[j] = coef[1]
        b[j] = target[:, j].mean()
    a, raw, clipped = _cap_vector(a, spectral_cap)
    return TrendAR(
        a=a,
        trend=trend,
        b=b,
        mu=mu,
        sd=sd,
        time_mu=time_mu,
        time_sd=time_sd,
        window=int(window),
        ridge=float(ridge),
        fit_record={
            "model": "trend_ar",
            "window": int(window),
            "ridge": float(ridge),
            "n_transitions": int(len(idx)),
            "raw_radius": raw,
            "clipped": bool(clipped),
            "spectral_radius": spectral_radius(np.diag(a)),
            "n_parameters": int(3 * n),
        },
    )


@dataclass
class CommonFactorAR:
    own_coef: np.ndarray
    own_mu: np.ndarray
    own_sd: np.ndarray
    factor_mu: np.ndarray
    factor_sd: np.ndarray
    b: np.ndarray
    factor_A: np.ndarray
    factor_b: np.ndarray
    k: int
    window: int
    ridge: float
    fit_record: dict[str, Any] = field(default_factory=dict)

    @property
    def n_parameters(self) -> int:
        return int(self.own_coef.size + self.factor_A.size + self.factor_b.size + len(self.b))

    def forecast_mean(self, x0: np.ndarray, f0: np.ndarray, horizon: int) -> np.ndarray:
        x = np.asarray(x0, dtype=np.float64).copy()
        f = np.asarray(f0, dtype=np.float64).copy()
        out = np.empty((int(horizon), x.size), dtype=np.float64)
        for step in range(int(horizon)):
            fs = (f - self.factor_mu) / self.factor_sd
            xs = (x - self.own_mu) / self.own_sd
            x = self.b + self.own_coef[:, 0] * xs + self.own_coef[:, 1:] @ fs
            f = self.factor_A @ f + self.factor_b
            out[step] = x
        return out


def fit_common_factor_ar(
    X: np.ndarray,
    F: np.ndarray,
    starts: np.ndarray,
    *,
    k: int,
    window: int,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
) -> CommonFactorAR:
    design, target, idx = _clean_transitions(X, starts)
    factors = np.asarray(F, dtype=np.float64)
    if factors.shape[0] != len(X):
        raise ValueError("factor scores must align with X")
    k = int(k)
    if k < 1 or k > factors.shape[1]:
        raise ValueError("k outside available factor scores")
    f_design = factors[idx, :k]
    factor_fit = fit_ridge_var(
        factors[:, :k],
        idx,
        ridge=float(ridge),
        spectral_cap=spectral_cap,
    )
    if factor_fit is None:
        raise ValueError("factor VAR could not be fitted")
    factor_A, factor_b, factor_record = factor_fit

    mu_x = design.mean(axis=0)
    sd_x = design.std(axis=0)
    sd_x[sd_x < 1e-8] = 1.0
    mu_f = f_design.mean(axis=0)
    sd_f = f_design.std(axis=0)
    sd_f[sd_f < 1e-8] = 1.0
    xs = (design - mu_x) / sd_x
    fs = (f_design - mu_f) / sd_f
    coef = np.empty((design.shape[1], 1 + k), dtype=np.float64)
    b = np.empty(design.shape[1], dtype=np.float64)
    for j in range(design.shape[1]):
        y = target[:, j] - target[:, j].mean()
        design2 = np.column_stack([xs[:, j], fs])
        beta = np.linalg.solve(
            design2.T @ design2 + float(ridge) * np.eye(1 + k),
            design2.T @ y,
        )
        coef[j] = beta
        b[j] = target[:, j].mean()
    return CommonFactorAR(
        own_coef=coef,
        own_mu=mu_x,
        own_sd=sd_x,
        factor_mu=mu_f,
        factor_sd=sd_f,
        b=b,
        factor_A=factor_A,
        factor_b=factor_b,
        k=k,
        window=int(window),
        ridge=float(ridge),
        fit_record={
            "model": "common_factor_ar",
            "window": int(window),
            "ridge": float(ridge),
            "k": k,
            "n_transitions": int(len(idx)),
            "n_parameters": int(coef.size + factor_A.size + factor_b.size + len(b)),
            "factor_spectral_radius": factor_record["realized_radius"],
        },
    )


def rollout_matched(model: MatchedOwnAR, x0: np.ndarray, eps: np.ndarray) -> np.ndarray:
    x = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], eps.shape[0], axis=0)
    out = np.empty((eps.shape[0], eps.shape[1], x.shape[1]), dtype=np.float64)
    for step in range(eps.shape[1]):
        x = model.a * x + model.b + eps[:, step]
        out[:, step] = x
    return out


def rollout_trend(model: TrendAR, x0: np.ndarray, t: int, eps: np.ndarray) -> np.ndarray:
    x = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], eps.shape[0], axis=0)
    out = np.empty((eps.shape[0], eps.shape[1], x.shape[1]), dtype=np.float64)
    for step in range(eps.shape[1]):
        time_value = float(t + step + 1)
        x = (
            model.b
            + model.a * (x - model.mu) / model.sd
            + model.trend * ((time_value - model.time_mu) / model.time_sd)
            + eps[:, step]
        )
        out[:, step] = x
    return out


def rollout_common_factor(
    model: CommonFactorAR, x0: np.ndarray, f0: np.ndarray, eps: np.ndarray
) -> np.ndarray:
    x = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], eps.shape[0], axis=0)
    f = np.repeat(np.asarray(f0, dtype=np.float64)[None, :], eps.shape[0], axis=0)
    out = np.empty((eps.shape[0], eps.shape[1], x.shape[1]), dtype=np.float64)
    for step in range(eps.shape[1]):
        fs = (f - model.factor_mu) / model.factor_sd
        xs = (x - model.own_mu) / model.own_sd
        x = model.b + model.own_coef[:, 0] * xs + fs @ model.own_coef[:, 1:].T + eps[:, step]
        f = f @ model.factor_A.T + model.factor_b
        out[:, step] = x
    return out


def rollout_persistence(x0: np.ndarray, eps: np.ndarray) -> np.ndarray:
    return np.asarray(x0, dtype=np.float64)[None, None, :] + np.cumsum(eps, axis=1)


def fit_window_own(
    X: np.ndarray,
    t: int,
    *,
    window: int,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
    min_transitions: int = 50,
) -> MatchedOwnAR | None:
    t = int(t)
    lo = max(1, t - int(window) + 1)
    starts = np.arange(lo, t, dtype=np.int64)
    if len(starts) < int(min_transitions):
        return None
    return fit_matched_own_ar(X, starts, window=window, ridge=ridge, spectral_cap=spectral_cap)


def fit_window_trend(
    X: np.ndarray,
    t: int,
    *,
    window: int,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
    min_transitions: int = 50,
) -> TrendAR | None:
    t = int(t)
    lo = max(1, t - int(window) + 1)
    starts = np.arange(lo, t, dtype=np.int64)
    if len(starts) < int(min_transitions):
        return None
    return fit_trend_ar(X, starts, window=window, ridge=ridge, spectral_cap=spectral_cap)


def fit_window_common(
    X: np.ndarray,
    F: np.ndarray,
    t: int,
    *,
    k: int,
    window: int,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
    min_transitions: int = 50,
) -> CommonFactorAR | None:
    t = int(t)
    lo = max(1, t - int(window) + 1)
    starts = np.arange(lo, t, dtype=np.int64)
    if len(starts) < int(min_transitions):
        return None
    return fit_common_factor_ar(
        X,
        F,
        starts,
        k=k,
        window=window,
        ridge=ridge,
        spectral_cap=spectral_cap,
    )


def coefficient_ablation(
    model: FrozenLinearModel, X: np.ndarray, t: int, *, window: int = 500
) -> FrozenLinearModel:
    """Zero B3's off-diagonal coefficients; this is not a refitted model."""
    return diagonal_copy(model, X, t, window=window)
