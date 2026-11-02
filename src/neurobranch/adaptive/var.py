"""Causal adaptive linear dynamics for the stability experiment.

All models are lag-one VARs on train-standardized traces.  A model fitted at a
prediction landmark uses only frames at or before that landmark; the forecast
then runs autonomously for the full horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def spectral_radius(A: np.ndarray) -> float:
    arr = np.asarray(A, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        return float("nan")
    return float(np.max(np.abs(np.linalg.eigvals(arr))))


def stabilize_matrix(A: np.ndarray, cap: float = 0.995) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale a lag-one operator to a requested spectral-radius cap."""
    arr = np.asarray(A, dtype=np.float64).copy()
    raw = spectral_radius(arr)
    if not np.isfinite(raw):
        raise ValueError("non-finite VAR operator")
    if raw <= cap or raw <= 0:
        return arr, {"raw_radius": raw, "realized_radius": raw, "scale": 1.0, "clipped": False}
    scale = float(cap / raw)
    return arr * scale, {
        "raw_radius": raw,
        "realized_radius": spectral_radius(arr * scale),
        "scale": scale,
        "clipped": True,
    }


def fit_ridge_var(
    X: np.ndarray,
    starts: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Weighted ridge VAR on transitions ``i -> i+1`` for ``i in starts``.

    The intercept is not penalized.  Features and targets are centered with
    the same weights, so the fit is invariant to a constant offset.
    """
    X = np.asarray(X, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    starts = starts[(starts >= 0) & (starts + 1 < len(X))]
    if len(starts) < 2:
        return None
    design = X[starts]
    target = X[starts + 1]
    finite = np.isfinite(design).all(axis=1) & np.isfinite(target).all(axis=1)
    design, target = design[finite], target[finite]
    starts = starts[finite]
    if len(starts) < 2:
        return None
    if weights is None:
        w = np.ones(len(starts), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)[finite]
        w = np.clip(w, 0.0, None)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return None
    w_sum = float(w.sum())
    mu_x = (w @ design) / w_sum
    mu_y = (w @ target) / w_sum
    xc = design - mu_x
    sd = np.sqrt((w @ (xc**2)) / w_sum)
    sd[sd < 1e-8] = 1.0
    xs = xc / sd
    yc = target - mu_y
    # Keep the penalty on the same scale as the validated unweighted VAR:
    # weights sum to the number of transitions rather than to one.
    w = w * (len(w) / w_sum)
    gram = xs.T @ (w[:, None] * xs)
    cross = xs.T @ (w[:, None] * yc)
    n = gram.shape[0]
    coef = np.linalg.solve(gram + float(ridge) * np.eye(n), cross)
    A = (coef / sd[:, None]).T
    b = mu_y - (mu_x / sd) @ coef
    record: dict[str, Any] = {
        "n_transitions": int(len(starts)),
        "first_transition": int(starts.min()),
        "last_transition": int(starts.max()),
        "ridge": float(ridge),
        "raw_radius": spectral_radius(A),
        "clipped": False,
        "scale": 1.0,
    }
    if spectral_cap is not None:
        A, stable = stabilize_matrix(A, cap=float(spectral_cap))
        record.update(stable)
    record["realized_radius"] = spectral_radius(A)
    return A, b, record


@dataclass
class FrozenVAR:
    A: np.ndarray
    b: np.ndarray
    ridge: float
    spectral_cap: float | None
    fit_record: dict[str, Any] = field(default_factory=dict)
    name: str = "frozen_var"

    @classmethod
    def fit(cls, X: np.ndarray, starts: np.ndarray, *, ridge: float = 1.0, spectral_cap: float | None = 0.995):
        fitted = fit_ridge_var(X, starts, ridge=ridge, spectral_cap=spectral_cap)
        if fitted is None:
            raise ValueError("not enough transitions for frozen VAR")
        A, b, record = fitted
        return cls(A=A, b=b, ridge=ridge, spectral_cap=spectral_cap, fit_record=record)

    def forecast(self, x0: np.ndarray, horizon: int) -> np.ndarray:
        x = np.asarray(x0, dtype=np.float64).copy()
        out = np.empty((horizon, x.size), dtype=np.float64)
        for step in range(horizon):
            x = self.A @ x + self.b
            out[step] = x
        return out

    @property
    def spectral_radius(self) -> float:
        return spectral_radius(self.A)


@dataclass
class RollingVAR:
    window: int
    ridge: float = 1.0
    spectral_cap: float | None = 0.995
    min_transitions: int = 50
    name: str = "rolling_var"

    def fit_at(self, X: np.ndarray, t: int) -> FrozenVAR | None:
        """Fit using transitions whose target is at or before ``t``."""
        lo = max(1, int(t) - int(self.window) + 1)
        hi = int(t)  # transition i -> i+1 with i <= t-1
        starts = np.arange(lo, hi, dtype=np.int64)
        if len(starts) < self.min_transitions:
            return None
        fitted = fit_ridge_var(X, starts, ridge=self.ridge, spectral_cap=self.spectral_cap)
        if fitted is None:
            return None
        A, b, record = fitted
        return FrozenVAR(
            A=A,
            b=b,
            ridge=self.ridge,
            spectral_cap=self.spectral_cap,
            fit_record={**record, "window": int(self.window), "landmark": int(t)},
            name="rolling_var",
        )


@dataclass
class EWMAVAR:
    halflife: float
    ridge: float = 1.0
    spectral_cap: float | None = 0.995
    min_transitions: int = 50
    name: str = "ewma_var"

    def fit_at(self, X: np.ndarray, t: int) -> FrozenVAR | None:
        """Exponentially weighted fit using transitions ending by ``t``."""
        hi = int(t)
        starts = np.arange(1, hi, dtype=np.int64)
        if len(starts) < self.min_transitions:
            return None
        age = (hi - 1) - starts
        weights = 0.5 ** (age / float(self.halflife))
        fitted = fit_ridge_var(
            X, starts, weights=weights, ridge=self.ridge, spectral_cap=self.spectral_cap
        )
        if fitted is None:
            return None
        A, b, record = fitted
        return FrozenVAR(
            A=A,
            b=b,
            ridge=self.ridge,
            spectral_cap=self.spectral_cap,
            fit_record={
                **record,
                "halflife": float(self.halflife),
                "landmark": int(t),
                "effective_sample_size": float(weights.sum() ** 2 / np.sum(weights**2)),
            },
            name="ewma_var",
        )
