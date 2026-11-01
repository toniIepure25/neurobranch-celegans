"""Forecastability diagnostics for held-out neural population windows."""

from __future__ import annotations

from typing import Any

import numpy as np

from neurobranch.diagnostics.common import (
    block_bootstrap_skill,
    dominant_frequency_and_power,
    forecast_matrix,
    make_forecast_starts,
    per_neuron_r2,
    per_neuron_skill,
    skill_vs_persistence,
)
from neurobranch.dynamics.context_models import StableLatentLDS
from neurobranch.dynamics.stable_var import StableVARModel
from neurobranch.preprocessing import split_starts


def _ridge_fit(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge with an unpenalized intercept in the last design column."""
    raw = design[:, :-1]
    mu = raw.mean(axis=0)
    sd = raw.std(axis=0)
    sd[sd < 1e-8] = 1.0
    xs = (raw - mu) / sd
    y_center = target - target.mean(axis=0)
    coef = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ y_center)
    intercept = target.mean(axis=0) - (mu / sd) @ coef
    return np.vstack([coef / sd[:, None], intercept])


def fit_neuron_ar(
    X: np.ndarray,
    starts: np.ndarray,
    *,
    history: int = 5,
    ridge: float = 1.0,
) -> np.ndarray:
    """One independent ridge AR model per neuron, direct one-step fit."""
    starts = np.asarray(starts, dtype=np.int64)
    design = np.stack([X[t - history + 1 : t + 1].ravel() for t in starts])
    design = np.column_stack([design, np.ones(len(starts))])
    target = X[starts + 1]
    return _ridge_fit(design, target, ridge)


def forecast_neuron_ar(
    histories: np.ndarray,
    coef: np.ndarray,
    *,
    horizon: int,
    history: int = 5,
) -> np.ndarray:
    """Recursive per-neuron AR forecasts; no future observation is used."""
    n_starts, _, n_neurons = histories.shape
    pred = np.empty((n_starts, horizon, n_neurons), dtype=np.float64)
    hist = histories[:, -history:, :].copy()
    for step in range(horizon):
        design = np.column_stack([hist.reshape(n_starts, -1), np.ones(n_starts)])
        x_next = design @ coef
        pred[:, step] = x_next
        hist = np.concatenate([hist[:, 1:], x_next[:, None]], axis=1)
    return pred


def forecast_persistence(histories: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(histories[:, -1:, :], horizon, axis=1)


def forecast_mean(histories: np.ndarray, horizon: int, mean: np.ndarray) -> np.ndarray:
    return np.repeat(mean[None, None, :], horizon, axis=1).repeat(len(histories), axis=0)


def forecast_trend(histories: np.ndarray, horizon: int) -> np.ndarray:
    hist = histories[:, -5:, :]
    slope = (hist[:, -1] - hist[:, 0]) / max(1, hist.shape[1] - 1)
    steps = np.arange(1, horizon + 1)[None, :, None]
    return hist[:, -1:, :] + steps * slope[:, None, :]


def fit_population_var(
    X: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    *,
    ridge: float = 1.0,
) -> StableVARModel:
    return StableVARModel.fit(
        X, y, starts, history=1, ridge=ridge, spectral_cap=0.995, noise_model="none"
    )


def forecast_var(model: StableVARModel, histories: np.ndarray, horizon: int) -> np.ndarray:
    n_starts, _, n_neurons = histories.shape
    pred = np.empty((n_starts, horizon, n_neurons), dtype=np.float64)
    current = histories[:, -1, :].copy()
    for step in range(horizon):
        current = np.asarray(model.predict_next(current[:, None, :]), dtype=np.float64).reshape(
            n_starts, n_neurons
        )
        pred[:, step] = current
    return pred


def fit_latent(
    X: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    *,
    train_mask: np.ndarray,
    latent_dim: int = 15,
    ridge: float = 1e-2,
) -> StableLatentLDS:
    return StableLatentLDS.fit(
        X,
        y,
        starts,
        train_mask=np.asarray(train_mask, dtype=bool),
        latent_dim=latent_dim,
        ridge=ridge,
        spectral_cap=0.995,
    )


def forecast_latent(
    model: StableLatentLDS, histories: np.ndarray, horizon: int
) -> np.ndarray:
    n_starts = len(histories)
    # Initial causal posterior mean per start, then a single batched linear
    # recursion instead of one Python rollout per start.
    z = np.stack([model.filter(h)["latent"][-1] for h in histories]).T
    pred = np.empty((n_starts, horizon, model.n_neurons), dtype=np.float64)
    for step in range(horizon):
        z = model.A @ z + model.b[:, None]
        pred[:, step] = model.decode(z.T)
    return pred


def _metrics_for_starts(
    pred: np.ndarray,
    truth: np.ndarray,
    persistence: np.ndarray,
) -> dict[str, Any]:
    err = (pred - truth) ** 2
    per_start_mse = err.mean(axis=(1, 2))
    per_start_persistence = ((persistence - truth) ** 2).mean(axis=(1, 2))
    ci = block_bootstrap_skill(per_start_mse, per_start_persistence, n_boot=400)
    step_mse = err.mean(axis=(0, 2))
    step_persistence = ((persistence - truth) ** 2).mean(axis=(0, 2))
    return {
        "mse": float(per_start_mse.mean()),
        "persistence_mse": float(per_start_persistence.mean()),
        "skill": skill_vs_persistence(float(per_start_mse.mean()), float(per_start_persistence.mean())),
        "skill_ci_low": ci["lo"],
        "skill_ci_high": ci["hi"],
        "r2": float(_safe_r2(truth, pred)),
        "step_mse": step_mse,
        "step_skill": np.array(
            [skill_vs_persistence(a, b) for a, b in zip(step_mse, step_persistence)]
        ),
        "per_neuron_skill": per_neuron_skill(pred, truth, persistence),
        "per_neuron_r2": per_neuron_r2(pred, truth),
        "per_start_mse": per_start_mse,
        "per_start_persistence": per_start_persistence,
    }


def _safe_r2(truth: np.ndarray, pred: np.ndarray) -> float:
    if truth.size == 0 or not np.isfinite(pred).all():
        return float("nan")
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - truth.mean(axis=(0, 1), keepdims=True)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def evaluate_models(
    data,
    *,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20, 50, 100),
    history: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit all simple baselines on train and evaluate identical test starts."""
    train_starts = split_starts(data, "train", history=history, horizon=1)
    if len(train_starts) == 0:
        raise ValueError("no training starts for forecastability")
    X, y = data.X, data.y_macro
    mean = X[train_starts].mean(axis=0)
    ar_coef = fit_neuron_ar(X, train_starts, history=history, ridge=1.0)
    var = fit_population_var(X, y, train_starts, ridge=1.0)
    latent = fit_latent(
        X, y, train_starts, train_mask=data.masks["train"], latent_dim=15, ridge=1e-2
    )

    summary: list[dict[str, Any]] = []
    per_neuron: list[dict[str, Any]] = []
    error_growth: list[dict[str, Any]] = []
    freq = dominant_frequency_and_power(X[data.masks["train"]], data.recording.fps)
    ar1 = np.array(
        [np.corrcoef(X[:-1, j], X[1:, j])[0, 1] for j in range(X.shape[1])]
    )

    for horizon in horizons:
        starts = make_forecast_starts(data, horizon=horizon, history=history)
        hist, truth = forecast_matrix(starts, X, history=history, horizon=horizon)
        persistence = forecast_persistence(hist, horizon)
        forecasts = {
            "persistence": persistence,
            "train_mean": forecast_mean(hist, horizon, mean),
            "local_trend": forecast_trend(hist, horizon),
            "neuron_ar5": forecast_neuron_ar(hist, ar_coef, horizon=horizon, history=history),
            "population_var": forecast_var(var, hist, horizon),
            "latent_lds": forecast_latent(latent, hist, horizon),
        }
        for name, pred in forecasts.items():
            m = _metrics_for_starts(pred, truth, persistence)
            summary.append(
                {
                    "model": name,
                    "worm_id": int(data.worm_id),
                    "horizon": int(horizon),
                    "horizon_seconds": float(horizon / data.recording.fps),
                    "n_starts": int(len(starts)),
                    "mse": m["mse"],
                    "persistence_mse": m["persistence_mse"],
                    "skill_vs_persistence": m["skill"],
                    "skill_ci_low": m["skill_ci_low"],
                    "skill_ci_high": m["skill_ci_high"],
                    "r2": m["r2"],
                    "per_neuron_median_skill": float(np.nanmedian(m["per_neuron_skill"])),
                }
            )
            for j, (skill, r2) in enumerate(zip(m["per_neuron_skill"], m["per_neuron_r2"])):
                per_neuron.append(
                    {
                        "model": name,
                        "worm_id": int(data.worm_id),
                        "horizon": int(horizon),
                        "neuron_index": j,
                        "neuron_name": data.predictor_names[j],
                        "identified": bool(any(ch.isalpha() for ch in data.predictor_names[j])),
                        "skill_vs_persistence": skill,
                        "r2": r2,
                        "autocorr1_train": float(ar1[j]),
                        "dominant_hz_train": float(freq["dominant_hz"][j]),
                        "low_frequency_fraction_train": float(freq["low_fraction"][j]),
                    }
                )
            for step, (mse, skill) in enumerate(zip(m["step_mse"], m["step_skill"]), start=1):
                error_growth.append(
                    {
                        "model": name,
                        "worm_id": int(data.worm_id),
                        "target_horizon": int(horizon),
                        "step": step,
                        "step_seconds": float(step / data.recording.fps),
                        "mse": float(mse),
                        "skill_vs_persistence": float(skill),
                    }
                )
    return summary, per_neuron, error_growth


def period_stability(
    data,
    *,
    horizon: int = 20,
    history: int = 5,
) -> list[dict[str, Any]]:
    train_starts = split_starts(data, "train", history=history, horizon=1)
    X, y = data.X, data.y_macro
    ar_coef = fit_neuron_ar(X, train_starts, history=history, ridge=1.0)
    var = fit_population_var(X, y, train_starts, ridge=1.0)
    latent = fit_latent(
        X, y, train_starts, train_mask=data.masks["train"], latent_dim=15, ridge=1e-2
    )
    starts = make_forecast_starts(data, horizon=horizon, history=history)
    chunks = np.array_split(starts, 3)
    rows = []
    for period, chunk in enumerate(chunks, start=1):
        if len(chunk) < 2:
            continue
        hist, truth = forecast_matrix(chunk, X, history=history, horizon=horizon)
        persistence = forecast_persistence(hist, horizon)
        forecasts = {
            "persistence": persistence,
            "neuron_ar5": forecast_neuron_ar(hist, ar_coef, horizon=horizon, history=history),
            "population_var": forecast_var(var, hist, horizon),
            "latent_lds": forecast_latent(latent, hist, horizon),
        }
        for name, pred in forecasts.items():
            rows.append(
                {
                    "model": name,
                    "worm_id": int(data.worm_id),
                    "period": period,
                    "horizon": int(horizon),
                    "n_starts": int(len(chunk)),
                    "start_frame_first": int(chunk[0]),
                    "start_frame_last": int(chunk[-1]),
                    "skill_vs_persistence": skill_vs_persistence(
                        float(np.mean((pred - truth) ** 2)),
                        float(np.mean((persistence - truth) ** 2)),
                    ),
                }
            )
    return rows


def noise_sensitivity(
    data,
    *,
    horizon: int = 20,
    history: int = 5,
    levels: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5),
    seed: int = 17,
) -> list[dict[str, Any]]:
    train_starts = split_starts(data, "train", history=history, horizon=1)
    X, y = data.X, data.y_macro
    ar_coef = fit_neuron_ar(X, train_starts, history=history, ridge=1.0)
    var = fit_population_var(X, y, train_starts, ridge=1.0)
    starts = make_forecast_starts(data, horizon=horizon, history=history)
    clean_hist, truth = forecast_matrix(starts, X, history=history, horizon=horizon)
    rng = np.random.default_rng(seed)
    rows = []
    for level in levels:
        hist = clean_hist + rng.normal(scale=level, size=clean_hist.shape)
        persistence = forecast_persistence(hist, horizon)
        forecasts = {
            "persistence": persistence,
            "neuron_ar5": forecast_neuron_ar(hist, ar_coef, horizon=horizon, history=history),
            "population_var": forecast_var(var, hist, horizon),
        }
        for name, pred in forecasts.items():
            rows.append(
                {
                    "model": name,
                    "worm_id": int(data.worm_id),
                    "horizon": int(horizon),
                    "noise_sd_standardized": float(level),
                    "skill_vs_persistence": skill_vs_persistence(
                        float(np.mean((pred - truth) ** 2)),
                        float(np.mean((persistence - truth) ** 2)),
                    ),
                }
            )
    return rows
