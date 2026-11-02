"""Diagnose and correct the causal calcium model's autonomous divergence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurobranch.adaptive.var import spectral_radius, stabilize_matrix
from neurobranch.diagnostics.observation import CausalCalciumLDS, _causal_calcium_transform


@dataclass
class CorrectedCalciumForecast:
    """Stable free rollout of the same fitted calcium model.

    The original rollout re-encodes each predicted calcium value as if it were
    a new high-pass drive, creating a positive feedback loop.  This version
    keeps the latent drive recursion and the calcium observation recursion
    separate: z' = A z + b, c' = (1-alpha)c + alpha(D z' + mean).
    """

    base: CausalCalciumLDS
    spectral_cap: float = 0.995
    version: str = "causal_calcium_lds_corrected_v1"

    def __post_init__(self) -> None:
        A, record = stabilize_matrix(np.asarray(self.base.A, dtype=np.float64), cap=self.spectral_cap)
        self.A = A
        self.stability = record

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        hist = np.asarray(history, dtype=np.float64)
        if hist.ndim != 2 or len(hist) < 2:
            raise ValueError("calcium forecast needs >=2 history frames")
        drive = _causal_calcium_transform(hist, float(self.base.alpha))
        z = np.asarray(self.base.pca.transform(drive), dtype=np.float64)[-1]
        c = hist[-1].copy()
        out = np.empty((int(horizon), hist.shape[1]), dtype=np.float64)
        for step in range(int(horizon)):
            z = self.A @ z + self.base.b
            drive_next = np.asarray(self.base.pca.inverse_transform(z[None]), dtype=np.float64)[0]
            c = (1.0 - float(self.base.alpha)) * c + float(self.base.alpha) * drive_next
            out[step] = c
        return out

    @property
    def alpha(self) -> float:
        return float(self.base.alpha)

    @property
    def observation_matrix(self) -> np.ndarray:
        return np.asarray(self.base.pca.components_.T, dtype=np.float64)

    @property
    def observation_mean(self) -> np.ndarray:
        return np.asarray(self.base.pca.mean_, dtype=np.float64)


def diagnose_fit(model: CausalCalciumLDS, data, *, history: int = 32, horizon: int = 100) -> dict[str, Any]:
    """Numerical diagnostics on one fitted old-style calcium model."""
    hist = np.asarray(data.X[-history:], dtype=np.float64)
    z = np.asarray(model.encode(hist), dtype=np.float64)
    A = np.asarray(model.A, dtype=np.float64)
    D = np.asarray(model.observation_matrix, dtype=np.float64)
    highpass_gain = float((1.0 - model.alpha) / model.alpha)
    with np.errstate(over="ignore", invalid="ignore"):
        old = model.rollout(data.X[-1], horizon, initial_history=hist)
        corrected = CorrectedCalciumForecast(model)
        new = corrected.forecast(hist, horizon)
    eig = np.linalg.eigvals(A)
    finite_old = np.isfinite(old).all(axis=1)
    finite_new = np.isfinite(new).all(axis=1)
    growth = np.linalg.norm(old, axis=1)
    return {
        "alpha": float(model.alpha),
        "latent_dim": int(model.latent_dim),
        "spectral_radius": spectral_radius(A),
        "max_eigenvalue_magnitude": float(np.max(np.abs(eig))),
        "min_eigenvalue_magnitude": float(np.min(np.abs(eig))),
        "observation_condition": float(np.linalg.cond(D)),
        "observation_sigma_max": float(np.linalg.svd(D, compute_uv=False)[0]),
        "highpass_gain": highpass_gain,
        "initial_latent_norm": float(np.linalg.norm(z[-1])),
        "old_finite_fraction": float(finite_old.mean()),
        "new_finite_fraction": float(finite_new.mean()),
        "old_max_abs": float(np.nanmax(np.abs(old))) if finite_old.any() else float("inf"),
        "new_max_abs": float(np.nanmax(np.abs(new))),
        "old_growth_ratio_20": float(growth[min(19, len(growth) - 1)] / max(1e-12, growth[0])),
        "old_growth_ratio_100": float(growth[-1] / max(1e-12, growth[0])),
        "new_growth_ratio_20": float(np.linalg.norm(new[19]) / max(1e-12, np.linalg.norm(new[0]))),
        "new_growth_ratio_100": float(np.linalg.norm(new[-1]) / max(1e-12, np.linalg.norm(new[0]))),
        "corrected_stability": corrected.stability,
    }
