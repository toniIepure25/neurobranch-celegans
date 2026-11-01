"""Versioned VAR with exact companion-matrix stability enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import LedoitWolf

from neurobranch.dynamics.models import Array, VARModel
from neurobranch.dynamics.stability import StabilityResult, stabilize_companion


@dataclass
class StableVARModel(VARModel):
    """VAR fitted without the legacy one-shot coefficient scaling.

    This is a new model class, not a mutation of a frozen Phase 6 checkpoint.
    The fit first estimates the unconstrained ridge VAR, then iteratively scales
    the AR coefficient block until the *actual* companion radius satisfies the
    requested cap.
    """

    stabilization: dict = field(default_factory=dict)
    name: str = "var_stable"

    @classmethod
    def fit(  # type: ignore[override]
        cls,
        X: Array,
        y: Array,
        starts: Array,
        *,
        history: int = 1,
        ridge: float = 1e-2,
        seed: int = 17,
        spectral_cap: float | None = 0.995,
        noise_model: str = "none",
        bootstrap_block: int = 5,
    ) -> StableVARModel:
        model = super().fit(
            X,
            y,
            starts,
            history=history,
            ridge=ridge,
            seed=seed,
            spectral_cap=None,
            noise_model="none",
            bootstrap_block=bootstrap_block,
        )
        diagnostics: dict = {"mode": "unconstrained"}
        if spectral_cap is not None:
            A, result = stabilize_companion(
                model.A,
                n_neurons=model.n_neurons,
                history=model.history_len,
                cap=float(spectral_cap),
            )
            model.A = A
            diagnostics = {"mode": "iterative_companion", **result.__dict__}
        model.spectral_cap = spectral_cap
        model.name = "var_stable"
        model.stabilization = diagnostics
        if noise_model != "none":
            model.noise_model = noise_model
            model.bootstrap_block = bootstrap_block
            model.native_stochastic = True
            residuals = X[starts + 1] - model.predict_next(
                np.stack([X[t - history + 1 : t + 1] for t in starts])
            )
            model.residuals = np.asarray(residuals, dtype=np.float64)
            model.noise_cov = np.asarray(
                LedoitWolf().fit(model.residuals).covariance_, dtype=np.float64
            )
        return model

    @property
    def realized_spectral_radius(self) -> float:
        return float(self.spectral_radius)

    def stability_record(self) -> dict:
        return {
            "model": "var_stable",
            "spectral_cap": self.spectral_cap,
            "realized_spectral_radius": self.realized_spectral_radius,
            **self.stabilization,
        }


def summarize_stability(result: StabilityResult) -> dict:
    return dict(result.__dict__)
