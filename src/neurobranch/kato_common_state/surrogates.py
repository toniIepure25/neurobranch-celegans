"""Split-safe common-factor-preserving temporal surrogates."""

from __future__ import annotations

from typing import Any

import numpy as np

from neurobranch.kato_common_state.factors import FactorModel
from neurobranch.kato_interactions.models import FrozenLinearModel, fit_var
from neurobranch.kato_interactions.surrogates import shuffle_window


def fit_common_state_surrogate(
    X: np.ndarray,
    t: int,
    *,
    factors: FactorModel,
    k: int,
    method: str,
    window: int = 500,
    block: int = 25,
    seed: int = 0,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
) -> tuple[FrozenLinearModel | None, dict[str, Any]]:
    """Fit B3 on a causal-window surrogate.

    ``common_preserved_residual_shift`` keeps the observed factor trajectory and
    shifts residual columns.  ``residual_preserved_factor_shift`` keeps residual
    columns and shifts factor columns.  All values are at or before ``t``.
    """
    t = int(t)
    lo = max(1, t - int(window) + 1)
    if t - lo < 50:
        return None, {"valid": False}
    values = np.asarray(X[lo : t + 1], dtype=np.float64)
    components = np.asarray(factors.pca.components_[: int(k)], dtype=np.float64)
    factor_values = factors.scores(values, int(k))
    mean = np.asarray(factors.pca.mean_, dtype=np.float64)
    common = factor_values @ components
    residual = values - mean - common
    diagnostics: dict[str, Any] = {
        "valid": True,
        "window": int(window),
        "landmark": t,
        "method": method,
        "k": int(k),
        "n": int(len(values)),
        "seed": int(seed),
    }
    if method == "circular_full":
        surrogate, diag = shuffle_window(values, method="circular", seed=seed)
        diagnostics.update(diag)
        diagnostics["common_factor_max_change"] = float(
            np.max(np.abs(factors.common(surrogate, k) - common))
        )
        diagnostics["residual_max_change"] = float(
            np.max(
                np.abs(
                    (surrogate - mean - factors.common(surrogate, k)) - residual
                )
            )
        )
    elif method == "block_full":
        surrogate, diag = shuffle_window(values, method="block", block=block, seed=seed)
        diagnostics.update(diag)
        diagnostics["common_factor_max_change"] = float(
            np.max(np.abs(factors.common(surrogate, k) - common))
        )
        diagnostics["residual_max_change"] = float(
            np.max(
                np.abs(
                    (surrogate - mean - factors.common(surrogate, k)) - residual
                )
            )
        )
    elif method == "common_preserved_residual_shift":
        residual_shift, diag = shuffle_window(residual, method="circular", seed=seed)
        # Independent column shifts need not remain orthogonal to the factor
        # subspace.  Project the shifted residual back out so the factor
        # trajectory is preserved exactly.
        residual_shift = residual_shift - (residual_shift @ components.T) @ components
        surrogate = mean + common + residual_shift
        diagnostics.update(diag)
        diagnostics["common_factor_max_change"] = float(
            np.max(np.abs(factors.common(surrogate, k) - common))
        )
        diagnostics["residual_max_change"] = float(
            np.max(
                np.abs(
                    (surrogate - mean - factors.common(surrogate, k)) - residual
                )
            )
        )
    elif method == "residual_preserved_factor_shift":
        factor_shift, diag = shuffle_window(factor_values, method="circular", seed=seed)
        surrogate = mean + factor_shift @ components + residual
        diagnostics.update(diag)
        diagnostics["common_factor_max_change"] = float(
            np.max(np.abs(factors.common(surrogate, k) - common))
        )
        diagnostics["residual_max_change"] = float(
            np.max(
                np.abs(
                    (surrogate - mean - factors.common(surrogate, k)) - residual
                )
            )
        )
    else:
        raise ValueError(f"unknown surrogate method {method!r}")

    starts = np.arange(0, len(surrogate) - 1, dtype=np.int64)
    model = fit_var(surrogate, starts, ridge=ridge, spectral_cap=spectral_cap)
    if model is not None:
        model.fit_record.update(
            {
                "split": "causal_window_surrogate",
                "surrogate_method": method,
                "surrogate_block": int(block),
                "surrogate_seed": int(seed),
                **diagnostics,
            }
        )
    return model, diagnostics
