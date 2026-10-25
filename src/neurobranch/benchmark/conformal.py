"""Validation-only conformal calibration for Phase 7 predictive intervals.

The quantile is fitted on held-out validation targets/animals and applied once to
the outer test set.  No test residual is ever used to choose the scale.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    level: float = 0.90,
) -> float:
    """Return the empirical ``level`` quantile of absolute normalised residuals."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    y = np.asarray(y_true, dtype=float)
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    ok = np.isfinite(y) & np.isfinite(m) & np.isfinite(s) & (s > 0)
    if ok.sum() < 5:
        return 1.0
    scores = np.abs(y[ok] - m[ok]) / s[ok]
    # "higher" is conservative for a finite calibration set.
    q = float(np.quantile(scores, level, method="higher"))
    return max(q, 1e-6)


def conformal_interval(
    mu: np.ndarray,
    sigma: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    width = float(q) * s
    return m - width, m + width
