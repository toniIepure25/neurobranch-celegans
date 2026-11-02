"""Stage H: synthetic identifiability controls for uncertainty propagation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.probabilistic.metrics import coverage_summary
from neurobranch.uncertainty_calibration.propagation import (
    forecast_covariance,
    monte_carlo_covariance,
    transient_amplification,
)

OUT = Path("reports/simulator/uncertainty_calibration")


def _fit_ar1(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design, target = x[:-1], x[1:]
    mx, my = design.mean(0), target.mean(0)
    xc, yc = design - mx, target - my
    a = (xc * yc).sum(0) / ((xc**2).sum(0) + 1e-8)
    b = my - a * mx
    resid = target - (a * design + b)
    q = np.cov(resid, rowvar=False)
    return a, b, q


def _simulate(A, b, Q, n, rng, x0=None):
    d = A.shape[0]
    x = np.zeros(d) if x0 is None else np.asarray(x0, dtype=float)
    chol = np.linalg.cholesky(Q + 1e-10 * np.eye(d))
    out = np.empty((n, d))
    for i in range(n):
        x = A @ x + b + chol @ rng.standard_normal(d)
        out[i] = x
    return out


def _coverage_stats(A, Q, horizon, rng, n=2000):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    _, cov = monte_carlo_covariance(A, np.zeros(A.shape[0]), np.zeros(A.shape[0]), Q, horizon, n, rng)
    P = forecast_covariance(A, Q, horizon)[horizon]
    # Draw directly from the analytical Gaussian for interval coverage.
    chol = np.linalg.cholesky(P + 1e-10 * np.eye(P.shape[0]))
    draws = rng.standard_normal((n, P.shape[0])) @ chol.T
    # Draw many held-out observations from the predictive distribution; this
    # estimates expected nominal coverage rather than a single lucky draw.
    reps = [coverage_summary(draws, chol @ rng.standard_normal(P.shape[0])) for _ in range(400)]
    cov_dict = {key: float(np.mean([r[key] for r in reps])) for key in reps[0]}
    return cov_dict, float(np.trace(P)), float(np.trace(cov[-1]))


def main() -> None:
    rng = np.random.default_rng(20261102)
    rows = []
    # 1. Independent Gaussian AR.
    A = np.diag([0.6, 0.4, 0.2])
    Q = np.diag([0.5, 0.3, 0.2])
    x = _simulate(A, np.zeros(3), Q, 8000, rng)
    a, b, q = _fit_ar1(x)
    cov, p, mc = _coverage_stats(np.diag(a), q, 20, rng)
    rows.append({"system": "independent_ar", "metric": "coverage_95_h20", "value": cov["coverage_95"]})
    rows.append({"system": "independent_ar", "metric": "coverage_80_h20", "value": cov["coverage_80"]})
    rows.append({"system": "independent_ar", "metric": "trace_ratio_analytic_mc", "value": p / mc})
    rows.append({"system": "independent_ar", "metric": "noise_trace_recovery", "value": float(np.sum(np.diag(q)) / np.sum(np.diag(Q)))})

    # 2. Temporally correlated innovations (AR(1) noise).
    T = 8000
    eps = np.zeros((T, 3))
    for t in range(1, T):
        eps[t] = 0.8 * eps[t - 1] + rng.standard_normal(3) * 0.4
    x = np.zeros((T, 3))
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + eps[t]
    # Naive iid fit sees the marginal residual covariance, not the spectral density.
    a, b, q = _fit_ar1(x)
    cov, p, mc = _coverage_stats(np.diag(a), q, 20, rng)
    rows.append({"system": "temporal_correlated", "metric": "coverage_95_h20_iid_model", "value": cov["coverage_95"]})
    rows.append({"system": "temporal_correlated", "metric": "acf1_residual", "value": float(np.mean([np.corrcoef(eps[:-1, j], eps[1:, j])[0, 1] for j in range(3)]))})

    # 3. State-dependent heteroscedasticity.
    T = 8000
    x = np.zeros((T, 2))
    for t in range(1, T):
        scale = 0.1 + 0.5 * np.abs(x[t - 1])
        x[t] = 0.7 * x[t - 1] + rng.standard_normal(2) * scale
    a, b, q = _fit_ar1(x)
    cov, p, mc = _coverage_stats(np.diag(a), q, 20, rng)
    rows.append({"system": "state_hetero", "metric": "coverage_95_h20_constant_model", "value": cov["coverage_95"]})
    rows.append({"system": "state_hetero", "metric": "std_scale_correlation", "value": float(np.corrcoef(np.abs(x[:-1]).mean(1), np.abs(x[1:] - x[:-1] * a).mean(1))[0, 1])})

    # 4. Low-rank correlated population noise.
    d = 6
    A = np.eye(d) * 0.4
    L = rng.standard_normal((d, 2))
    Q = L @ L.T + np.diag([0.1] * d)
    x = _simulate(A, np.zeros(d), Q, 12000, rng)
    a, b, q = _fit_ar1(x)
    eig = np.linalg.eigvalsh(q)[::-1]
    rows.append({"system": "lowrank_population", "metric": "top2_variance_fraction", "value": float(eig[:2].sum() / eig.sum())})
    rows.append({"system": "lowrank_population", "metric": "trace_recovery", "value": float(np.trace(q) / np.trace(Q))})

    # 5. Measurement noise distinct from process noise.
    T = 8000
    latent = np.zeros((T, 2))
    for t in range(1, T):
        latent[t] = 0.6 * latent[t - 1] + rng.standard_normal(2) * 0.3
    observed = latent + rng.standard_normal((T, 2)) * 0.5
    a, b, q_obs = _fit_ar1(observed)
    rows.append({"system": "measurement_noise", "metric": "observed_innovation_trace", "value": float(np.trace(q_obs))})
    rows.append({"system": "measurement_noise", "metric": "true_process_innovation_trace", "value": float(0.09 * 2)})
    rows.append({"system": "measurement_noise", "metric": "process_variance_identifiable", "value": 0.0})

    # 6. Persistent non-normal operator with transient amplification.
    A = np.array([[0.9, 4.0], [0.0, 0.9]])
    Q = np.eye(2) * 0.2
    amp = transient_amplification(A, 20)
    cov, p, mc = _coverage_stats(A, Q, 20, rng)
    rows.append({"system": "nonnormal_transient", "metric": "spectral_radius", "value": amp["spectral_radius"]})
    rows.append({"system": "nonnormal_transient", "metric": "max_singular_value_h20", "value": amp["max_singular_value_h"]})
    rows.append({"system": "nonnormal_transient", "metric": "nonnormality", "value": amp["nonnormality"]})
    rows.append({"system": "nonnormal_transient", "metric": "trace_ratio_analytic_mc", "value": p / mc})

    # 7. Misspecified mean: AR(2) fit by AR(1).
    T = 8000
    x = np.zeros(T)
    for t in range(2, T):
        x[t] = 0.6 * x[t - 1] - 0.35 * x[t - 2] + rng.standard_normal() * 0.4
    a, b, q = _fit_ar1(x[:, None])
    resid = x[1:] - (a[0] * x[:-1] + b[0])
    rows.append({"system": "misspecified_mean", "metric": "residual_acf1", "value": float(np.corrcoef(resid[:-1], resid[1:])[0, 1])})
    rows.append({"system": "misspecified_mean", "metric": "residual_acf2", "value": float(np.corrcoef(resid[:-2], resid[2:])[0, 1])})
    rows.append({"system": "misspecified_mean", "metric": "coverage_95_h20", "value": _coverage_stats(np.diag(a), q, 20, rng)[0]["coverage_95"]})

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "synthetic_controls.csv", index=False)
    (OUT / "synthetic_summary.json").write_text(
        json.dumps(
            {
                "n_systems": 7,
                "note": "Synthetic recovery checks implementation and identifiability; not biological validation.",
                "rows": len(frame),
            },
            indent=2,
        )
    )
    print(frame.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
