"""Synthetic ground-truth checks for common-factor versus coupling controls."""

from __future__ import annotations

from typing import Any

import numpy as np

from neurobranch.kato_common_state.factors import fit_factor_model
from neurobranch.kato_common_state.models import (
    fit_window_common,
    fit_window_own,
)
from neurobranch.kato_interactions.models import fit_window


def simulate_system(
    scenario: str,
    seed: int,
    *,
    n_frames: int = 2400,
    n_neurons: int = 20,
    n_factors: int = 3,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    burn = 200
    total = int(n_frames) + burn
    d = int(n_neurons)
    k = int(n_factors)
    loadings = rng.normal(size=(d, k))
    loadings /= np.maximum(np.linalg.norm(loadings, axis=0, keepdims=True), 1e-8)
    factor_ar = np.diag([0.92, 0.85, 0.78][:k])
    factor_noise = 0.35
    f = np.zeros(k, dtype=np.float64)
    x = np.zeros(d, dtype=np.float64)
    direct = 0.55 * np.eye(d) + 0.32 * np.roll(np.eye(d), 1, axis=1)
    direct = direct / max(float(np.max(np.abs(np.linalg.eigvals(direct)))), 1e-8) * 0.82
    use_common = scenario in {"common", "coupled_common", "time_varying_common"}
    use_direct = scenario in {"coupled", "coupled_common"}
    series = np.empty((total, d), dtype=np.float64)
    factor_series = np.empty((total, k), dtype=np.float64)
    for t in range(total):
        if use_common:
            f = factor_ar @ f + factor_noise * rng.normal(size=k)
            current_loadings = loadings.copy()
            if scenario == "time_varying_common":
                scale = 1.0 + 0.45 * np.sin(2.0 * np.pi * t / 160.0)
                rotation = np.array(
                    [
                        [np.cos(0.15 * np.sin(t / 200.0)), -np.sin(0.15 * np.sin(t / 200.0))],
                        [np.sin(0.15 * np.sin(t / 200.0)), np.cos(0.15 * np.sin(t / 200.0))],
                    ]
                )
                current_loadings[:, :2] = current_loadings[:, :2] @ rotation
                current_loadings *= scale
            common = current_loadings @ f
        else:
            f = np.zeros(k, dtype=np.float64)
            common = np.zeros(d, dtype=np.float64)
        x = (
            direct @ x + 0.30 * rng.normal(size=d)
            if use_direct
            else 0.60 * x + 0.30 * rng.normal(size=d)
        )
        series[t] = common + x
        factor_series[t] = f
    return {
        "scenario": scenario,
        "seed": int(seed),
        "X": series[burn:],
        "factors_true": factor_series[burn:],
        "direct_A": direct if use_direct else np.zeros_like(direct),
        "loadings": loadings,
        "use_common": bool(use_common),
        "use_direct": bool(use_direct),
    }


def evaluate_system(
    system: dict[str, Any],
    *,
    train_frames: int = 1600,
    window: int = 1200,
    horizon: int = 20,
    stride: int = 20,
    factor_ks: tuple[int, ...] = (3, 10),
) -> dict[str, Any]:
    X = np.asarray(system["X"], dtype=np.float64)
    train = X[: int(train_frames)]
    factor_model = fit_factor_model(train, n_components=10, seed=0)
    all_factors = factor_model.scores(X, 10)
    landmarks = np.arange(int(train_frames) + 2, len(X) - horizon, int(stride), dtype=np.int64)
    rows = []
    for t in landmarks:
        truth = X[t + 1 : t + horizon + 1]
        x0 = X[t]
        b1 = fit_window_own(X, t, window=200, ridge=1.0)
        b3 = fit_window(X, t, window=window, ridge=1.0, spectral_cap=0.995)
        if b1 is None or b3 is None:
            continue
        f0 = all_factors[t]
        b5 = {}
        for k in factor_ks:
            b5[k] = fit_window_common(
                X,
                all_factors,
                t,
                k=k,
                window=window,
                ridge=1.0,
                spectral_cap=0.995,
            )
        b1_forecast = b1.forecast_mean(x0, horizon)
        b3_forecast = b3.forecast_mean(x0, horizon)
        row = {
            "landmark": int(t),
            "mse_b1": float(np.mean((b1_forecast - truth) ** 2)),
            "mse_b3": float(np.mean((b3_forecast - truth) ** 2)),
            "b3_offdiag_norm": float(np.linalg.norm(b3.A - np.diag(np.diag(b3.A)), "fro")),
        }
        for k, model in b5.items():
            if model is None:
                row[f"mse_b5_{k}"] = float("nan")
                continue
            forecast = model.forecast_mean(x0, f0[:k], horizon)
            row[f"mse_b5_{k}"] = float(np.mean((forecast - truth) ** 2))
        rows.append(row)
    if not rows:
        raise RuntimeError("no synthetic landmarks could be scored")
    frame = {
        key: float(np.nanmean([row[key] for row in rows]))
        for key in rows[0]
        if key != "landmark"
    }
    for k in factor_ks:
        frame[f"b3_minus_b5_{k}_mse_gain"] = float(frame[f"mse_b5_{k}"] - frame["mse_b3"])
        frame[f"b5_{k}_minus_b1_mse_gain"] = float(frame["mse_b1"] - frame[f"mse_b5_{k}"])
    frame["true_offdiag_norm"] = float(
        np.linalg.norm(system["direct_A"] - np.diag(np.diag(system["direct_A"])), "fro")
    )
    frame.update(
        {
            "scenario": system["scenario"],
            "seed": int(system["seed"]),
            "n_landmarks": int(len(rows)),
            "factor_variance_10": float(np.sum(factor_model.pca.explained_variance_ratio_)),
        }
    )
    return frame
