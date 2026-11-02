"""Small causal linear operators used by the Kato interaction audit.

The full operator is a lag-one VAR.  The diagonal ablation keeps the fitted
own-neuron coefficients and only removes off-diagonal propagation; its intercept
is recomputed from the same transition sample.  This is a mathematical
ablation, not a biological lesion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neurobranch.adaptive.var import fit_ridge_var, spectral_radius


@dataclass
class FrozenLinearModel:
    """Stable lag-one operator with an unpenalized intercept."""

    A: np.ndarray
    b: np.ndarray
    fit_record: dict[str, Any] = field(default_factory=dict)

    def forecast_mean(self, x0: np.ndarray, horizon: int) -> np.ndarray:
        x = np.asarray(x0, dtype=np.float64).copy()
        out = np.empty((int(horizon), x.size), dtype=np.float64)
        for step in range(int(horizon)):
            x = self.A @ x + self.b
            out[step] = x
        return out

    def impulse(self, delta: np.ndarray, horizon: int) -> np.ndarray:
        """Linear impulse response ``A**h @ delta`` for h=1..horizon."""
        delta = np.asarray(delta, dtype=np.float64)
        out = np.empty((int(horizon), delta.size), dtype=np.float64)
        z = delta.copy()
        for step in range(int(horizon)):
            z = self.A @ z
            out[step] = z
        return out

    @property
    def spectral_radius(self) -> float:
        return spectral_radius(self.A)

    @property
    def offdiagonal_energy(self) -> float:
        return float(np.linalg.norm(self.A - np.diag(np.diag(self.A)), "fro"))

    @property
    def diagonal_energy(self) -> float:
        return float(np.linalg.norm(np.diag(self.A)))


def _transition_means(X: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design, target = X[starts], X[starts + 1]
    good = np.isfinite(design).all(axis=1) & np.isfinite(target).all(axis=1)
    design, target = design[good], target[good]
    return design.mean(axis=0), target.mean(axis=0)


def diagonalize_operator(A: np.ndarray) -> np.ndarray:
    return np.diag(np.diag(np.asarray(A, dtype=np.float64)))


def diagonal_copy(model: FrozenLinearModel, X: np.ndarray, t: int, *, window: int = 500) -> FrozenLinearModel:
    """Remove cross-neuron propagation from a fitted operator, keeping its diagonal."""
    t = int(t)
    lo = max(1, t - int(window) + 1)
    starts = np.arange(lo, t, dtype=np.int64)
    starts = starts[(starts >= 0) & (starts + 1 < len(X))]
    if len(starts) < 1:
        raise ValueError("no transitions for diagonal copy")
    mu_x, mu_y = _transition_means(np.asarray(X, dtype=np.float64), starts)
    A = diagonalize_operator(model.A)
    b = mu_y - A @ mu_x
    record = {
        **model.fit_record,
        "diagonal": True,
        "diagonalized_from": "fitted_population_operator",
        "offdiagonal_energy": 0.0,
        "realized_radius": spectral_radius(A),
    }
    return FrozenLinearModel(A=A, b=b, fit_record=record)



def fit_var(
    X: np.ndarray,
    starts: np.ndarray,
    *,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
    diagonal: bool = False,
) -> FrozenLinearModel | None:
    """Fit a ridge VAR on explicit causal transition starts."""
    X = np.asarray(X, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    starts = starts[(starts >= 0) & (starts + 1 < len(X))]
    if len(starts) < 2:
        return None
    fitted = fit_ridge_var(X, starts, ridge=float(ridge), spectral_cap=spectral_cap)
    if fitted is None:
        return None
    A, b, record = fitted
    if diagonal:
        mu_x, mu_y = _transition_means(X, starts)
        A = diagonalize_operator(A)
        # A diagonal operator has diagonal eigenvalues; retain the fitted cap.
        raw = spectral_radius(A)
        if spectral_cap is not None and raw > float(spectral_cap):
            A = A * (float(spectral_cap) / max(raw, 1e-12))
        b = mu_y - A @ mu_x
        record = {
            **record,
            "diagonal": True,
            "raw_radius": float(raw),
            "realized_radius": spectral_radius(A),
            "offdiagonal_energy": 0.0,
        }
    else:
        record = {
            **record,
            "diagonal": False,
            "offdiagonal_energy": float(np.linalg.norm(A - np.diag(np.diag(A)), "fro")),
        }
    return FrozenLinearModel(A=A, b=b, fit_record=record)


def fit_window(
    X: np.ndarray,
    t: int,
    *,
    window: int = 500,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
    min_transitions: int = 50,
    diagonal: bool = False,
) -> FrozenLinearModel | None:
    """Fit using transitions whose targets end at or before ``t``."""
    t = int(t)
    lo = max(1, t - int(window) + 1)
    starts = np.arange(lo, t, dtype=np.int64)
    if len(starts) < int(min_transitions):
        return None
    model = fit_var(
        X,
        starts,
        ridge=ridge,
        spectral_cap=spectral_cap,
        diagonal=diagonal,
    )
    if model is not None:
        model.fit_record.update({"window": int(window), "landmark": t, "split": "causal_window"})
    return model


def fit_window_diagonal(
    X: np.ndarray,
    t: int,
    *,
    window: int = 500,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
    min_transitions: int = 50,
) -> FrozenLinearModel | None:
    return fit_window(
        X,
        t,
        window=window,
        ridge=ridge,
        spectral_cap=spectral_cap,
        min_transitions=min_transitions,
        diagonal=True,
    )
