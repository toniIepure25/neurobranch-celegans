"""Stage A/B diagnostics: variance propagation and innovation validity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from neurobranch.adaptive.evaluation import evaluation_landmarks
from neurobranch.adaptive.var import RollingVAR, spectral_radius
from neurobranch.diagnostics.common import load_nostim
from neurobranch.probabilistic.baselines import IndependentARModel
from neurobranch.probabilistic.residual import (
    adaptive_prequential_residuals,
    independent_ar_residuals,
    regularized_covariance,
)
from neurobranch.uncertainty_calibration.models import state_features
from neurobranch.uncertainty_calibration.propagation import (
    forecast_covariance,
    monte_carlo_covariance,
    transient_amplification,
)

OUT = Path("reports/simulator/uncertainty_calibration")
OUT.mkdir(parents=True, exist_ok=True)


def _acf_abs(values: np.ndarray, lag: int) -> float:
    vals = []
    for j in range(values.shape[1]):
        a, b = values[:-lag, j], values[lag:, j]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            vals.append(abs(float(np.corrcoef(a, b)[0, 1])))
    return float(np.median(vals))


def _whiten(values: np.ndarray) -> tuple[np.ndarray, float]:
    phi = []
    for j in range(values.shape[1]):
        a, b = values[:-1, j], values[1:, j]
        phi.append(float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 else 0.0)
    phi = np.asarray(phi)
    return values[1:] - phi[None, :] * values[:-1], float(np.median(phi))


def _state_variance_holdout(X: np.ndarray, residuals, train_end: int, window: int = 10) -> tuple[float, float]:
    mask = residuals.targets <= int(train_end)
    targets = residuals.targets[mask]
    values = residuals.values[mask]
    feats = np.stack([state_features(X, int(k) - 1, window) for k in targets])
    n, d, p = feats.shape
    flat = feats.reshape(n * d, p)
    mean, std = flat.mean(0), flat.std(0)
    std = np.where(std > 1e-12, std, 1.0)
    z = ((flat - mean) / std).reshape(n, d, p)
    split = int(0.7 * n)
    r2_log, r2_var = [], []
    for j in range(d):
        model = Ridge(alpha=1.0).fit(z[:split, j], np.log(np.maximum(values[:split, j] ** 2, 1e-6)))
        pred = model.predict(z[split:, j])
        y = values[split:, j] ** 2
        if np.var(y) > 1e-12:
            r2_log.append(1 - np.mean((np.log(y + 1e-6) - pred) ** 2) / np.var(np.log(y + 1e-6)))
            r2_var.append(1 - np.mean((y - np.exp(np.clip(pred, -20, 20))) ** 2) / np.var(y))
    return float(np.median(r2_log)), float(np.median(r2_var))


def _mean_predictability(X: np.ndarray, residuals, train_end: int) -> float:
    mask = residuals.targets <= int(train_end)
    states, values = residuals.states[mask], residuals.values[mask]
    split = int(0.7 * len(states))
    xm, xs = states[:split].mean(0), states[:split].std(0)
    xs = np.where(xs > 1e-12, xs, 1.0)
    z = (states - xm) / xs
    model = Ridge(alpha=10.0).fit(z[:split], values[:split])
    pred = model.predict(z[split:])
    r2 = []
    for j in range(values.shape[1]):
        denom = np.var(values[split:, j])
        if denom > 1e-12:
            r2.append(1 - np.mean((values[split:, j] - pred[:, j]) ** 2) / denom)
    return float(np.median(r2))


def _cross_cov_stability(values: np.ndarray) -> tuple[float, float]:
    half = len(values) // 2
    c1 = np.corrcoef(values[:half], rowvar=False)
    c2 = np.corrcoef(values[half:], rowvar=False)
    iu = np.triu_indices(c1.shape[0], 1)
    corr = float(np.corrcoef(c1[iu], c2[iu])[0, 1])
    ratio = float(np.trace(np.cov(values[half:], rowvar=False)) / np.trace(np.cov(values[:half], rowvar=False)))
    return corr, ratio


def _series_metrics(session: str, model: str, rs, train_end: int, X: np.ndarray) -> list[dict]:
    rows = []
    values = rs.values
    for lead in (1, 5, 20, 50):
        mask = rs.lead == lead
        if not mask.any():
            continue
        sub = values[mask]
        rows.append(
            {
                "worm_id": session,
                "model": model,
                "lead": lead,
                "n": int(mask.sum()),
                "trace_cov": float(np.trace(np.cov(sub, rowvar=False))),
                "mean_squared": float(np.mean(sub**2)),
            }
        )
    acf1, acf5, acf10 = (_acf_abs(values, k) for k in (1, 5, 10))
    whitened, phi = _whiten(values)
    r2_log, r2_var = _state_variance_holdout(X, rs, train_end)
    cross_corr, var_ratio = _cross_cov_stability(values)
    rows.append(
        {
            "worm_id": session,
            "model": model,
            "lead": -1,
            "n": int(len(values)),
            "trace_cov": float(np.trace(np.cov(values, rowvar=False))),
            "trace_cov_whitened": float(np.trace(np.cov(whitened, rowvar=False))),
            "acf1": acf1,
            "acf5": acf5,
            "acf10": acf10,
            "whitened_acf1": _acf_abs(whitened, 1),
            "phi_median": phi,
            "cross_cov_split_corr": cross_corr,
            "variance_second_half_ratio": var_ratio,
            "mean_r2_from_state": _mean_predictability(X, rs, train_end),
            "state_var_log_r2": r2_log,
            "state_var_r2": r2_var,
        }
    )
    return rows


def main() -> None:
    cfg, data, _ = load_nostim()
    all_series, all_ops, all_prop = [], [], []
    for d in data:
        sid = str(d.worm_id)
        train_end = int(d.split_bounds["train"][1] - 1)
        landmarks = evaluation_landmarks(d, horizon_max=100, stride=10)
        adaptive = adaptive_prequential_residuals(d, window=500, block=10)
        indar = independent_ar_residuals(d, window=200, block=10)
        all_series.extend(_series_metrics(sid, "adaptive_var", adaptive, train_end, d.X))
        all_series.extend(_series_metrics(sid, "independent_ar", indar, train_end, d.X))
        all_series.extend(_series_metrics(sid, "adaptive_var_block50", adaptive_prequential_residuals(d, window=500, block=50), train_end, d.X))
        all_series.extend(_series_metrics(sid, "independent_ar_block50", independent_ar_residuals(d, window=200, block=50), train_end, d.X))

        for k, t in enumerate(landmarks):
            t = int(t)
            f = RollingVAR(window=500).fit_at(d.X, t)
            ar = IndependentARModel(window=200).fit_at(d.X, t)
            if f is None or ar is None:
                continue
            # Re-select lead 1 exactly from the causal subset.
            mask = (adaptive.targets <= t) & (adaptive.lead == 1)
            r1 = adaptive.values[mask][-300:]
            rall = adaptive.values[adaptive.targets <= t][-1000:]
            Q1, _ = regularized_covariance(r1, method="ledoit")
            Qa, _ = regularized_covariance(rall, method="ledoit")
            Q1_emp = np.cov(r1, rowvar=False)
            Qa_emp = np.cov(rall, rowvar=False)
            P1 = forecast_covariance(f.A, Q1_emp, 100)
            Pa = forecast_covariance(f.A, Qa_emp, 100)
            amask = (indar.targets <= t) & (indar.lead == 1)
            ri = indar.values[amask][-300:]
            Qi = np.diag(np.maximum(ri.var(axis=0, ddof=1), 1e-8))
            Pi = forecast_covariance(np.diag(ar.a), Qi, 100)
            amp = transient_amplification(f.A, 20)
            all_ops.append(
                {
                    "worm_id": sid,
                    "landmark": t,
                    "radius_adaptive": spectral_radius(f.A),
                    "radius_independent": float(np.max(np.abs(ar.a))),
                    "norm_A20": amp["max_singular_value_h"],
                    "nonnormality": amp["nonnormality"],
                    "trace_Q_lead1": float(np.trace(Q1_emp)),
                    "trace_Q_all": float(np.trace(Qa_emp)),
                    "trace_Q_lead1_ledoit": float(np.trace(Q1)),
                    "trace_Q_all_ledoit": float(np.trace(Qa)),
                    "trace_Q_independent_lead1": float(np.trace(Qi)),
                    "trace_P20_lead1": float(np.trace(P1[20])),
                    "trace_P20_all": float(np.trace(Pa[20])),
                    "trace_P20_independent": float(np.trace(Pi[20])),
                    "trace_P100_lead1": float(np.trace(P1[100])),
                    "trace_P100_all": float(np.trace(Pa[100])),
                    "trace_P100_independent": float(np.trace(Pi[100])),
                }
            )
            if k % 4 == 0:
                # Monte Carlo and block-bootstrap checks at H20.
                _, cov_mc1 = monte_carlo_covariance(f.A, f.b, d.X[t], Q1_emp, 20, 1200, np.random.default_rng(1000 + t))
                _, cov_mca = monte_carlo_covariance(f.A, f.b, d.X[t], Qa_emp, 20, 1200, np.random.default_rng(2000 + t))
                # Block bootstrap all-lead and lead1.
                from neurobranch.probabilistic.stochastic import Innovation, ensemble_rollout

                centered1 = r1 - r1.mean(0)
                centereda = rall - rall.mean(0)
                ens1 = ensemble_rollout(
                    "var", f, d.X[t], 20, 500, Innovation("block", pool=centered1, block=10, metadata={}),
                    np.random.default_rng(3000 + t),
                )
                ensa = ensemble_rollout(
                    "var", f, d.X[t], 20, 500, Innovation("block", pool=centereda, block=10, metadata={}),
                    np.random.default_rng(4000 + t),
                )
                all_prop.append(
                    {
                        "worm_id": sid,
                        "landmark": t,
                        "trace_analytical_lead1": float(np.trace(P1[20])),
                        "trace_analytical_all": float(np.trace(Pa[20])),
                        "trace_mc_gaussian_lead1": float(np.trace(cov_mc1[-1])),
                        "trace_mc_gaussian_all": float(np.trace(cov_mca[-1])),
                        "trace_block_lead1": float(np.trace(np.cov(ens1[:, -1], rowvar=False))),
                        "trace_block_all": float(np.trace(np.cov(ensa[:, -1], rowvar=False))),
                        "trace_independent": float(np.trace(Pi[20])),
                        "observed_h20_mse": float(np.mean((f.forecast(d.X[t], 20)[-1] - d.X[t + 20]) ** 2)),
                        "observed_h1_mse": float(np.mean((f.forecast(d.X[t], 1)[-1] - d.X[t + 1]) ** 2)),
                    }
                )
    pd.DataFrame(all_series).to_csv(OUT / "innovation_validation.csv", index=False)
    pd.DataFrame(all_ops).to_csv(OUT / "operator_diagnostics.csv", index=False)
    pd.DataFrame(all_prop).to_csv(OUT / "variance_propagation.csv", index=False)
    summary = {
        "n_sessions": len(data),
        "lead1_vs_all": {
            "adaptive_trace_lead1_median": float(pd.DataFrame(all_ops).trace_Q_lead1.median()),
            "adaptive_trace_all_median": float(pd.DataFrame(all_ops).trace_Q_all.median()),
            "independent_trace_lead1_median": float(pd.DataFrame(all_ops).trace_Q_independent_lead1.median()),
        },
        "propagation": pd.DataFrame(all_prop).median(numeric_only=True).to_dict(),
        "innovation": pd.DataFrame(all_series).query("lead == -1").groupby("model").median(numeric_only=True).to_dict(),
    }
    (OUT / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("wrote diagnostics", len(all_series), len(all_ops), len(all_prop))


if __name__ == "__main__":
    main()
