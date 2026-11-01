"""Synthetic known-truth controls for observation-aware diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from neurobranch.diagnostics.common import random_stable_system
from neurobranch.diagnostics.observation import (
    CausalCalciumLDS,
    SecondOrderLatentLDS,
    _causal_calcium_transform,
)
from neurobranch.dynamics.context_models import StableLatentLDS


def _r2(truth: np.ndarray, pred: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - truth.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _skill(mse: float, base: float) -> float:
    return 1.0 - mse / base if base > 0 else float("nan")


def _linear_alignment_r2(true_z: np.ndarray, est_z: np.ndarray) -> tuple[float, float]:
    """Symmetric linear-map R2; invariant to rotation/sign/permutation."""
    A = np.column_stack([true_z, np.ones(len(true_z))])
    coef = np.linalg.lstsq(A, est_z, rcond=None)[0]
    pred_est = A @ coef
    A2 = np.column_stack([est_z, np.ones(len(est_z))])
    coef2 = np.linalg.lstsq(A2, true_z, rcond=None)[0]
    pred_true = A2 @ coef2
    return float(_r2(est_z, pred_est)), float(_r2(true_z, pred_true))


def _fit_models(X: np.ndarray, train_mask: np.ndarray, train_starts: np.ndarray, k: int, seed: int = 17):
    labels = np.random.default_rng(seed).integers(0, 4, size=len(X))
    ar1 = StableLatentLDS.fit(
        X,
        labels,
        train_starts,
        train_mask=train_mask,
        latent_dim=k,
        ridge=1e-2,
        spectral_cap=0.995,
    )
    ar2 = SecondOrderLatentLDS.fit(
        X, train_mask, train_starts, latent_dim=k, ridge=1e-2, spectral_cap=0.995
    )
    calcium = CausalCalciumLDS.fit(
        X, train_mask, train_starts, latent_dim=k, ridge=1e-2, spectral_cap=0.995
    )
    return ar1, ar2, calcium


def _forecast_ar1(model: StableLatentLDS, hist: np.ndarray, horizon: int) -> np.ndarray:
    out = []
    for h in hist:
        out.append(model.rollout(np.zeros(model.n_neurons), horizon, initial_history=h)[1:])
    return np.stack(out)


def _forecast_generic(model, hist: np.ndarray, horizon: int) -> np.ndarray:
    out = []
    for h in hist:
        out.append(model.rollout(np.zeros(model.n_neurons), horizon, initial_history=h)[1:])
    return np.stack(out)


def _true_filter_c(z: np.ndarray, alpha: float) -> np.ndarray:
    c = np.zeros_like(z)
    c[0] = z[0]
    for t in range(1, len(z)):
        c[t] = (1.0 - alpha) * c[t - 1] + alpha * z[t]
    return c


def _perturbation_rows(
    system,
    true_z: np.ndarray,
    models: dict[str, Any],
    X: np.ndarray,
    frames: np.ndarray,
    *,
    neuron: int = 0,
    amplitude: float = 0.5,
    horizon: int = 20,
) -> list[dict[str, Any]]:
    D = system.D
    delta_obs = np.zeros(system.n_neurons)
    delta_obs[neuron] = amplitude
    delta_z = np.linalg.pinv(D) @ delta_obs
    alpha = system.alpha
    true_c = _true_filter_c(true_z, alpha)
    rows = []
    for frame in frames:
        # Known-truth response to an observation-space shift at this frame.
        z0 = true_z[frame].copy()
        c0 = true_c[frame].copy()
        base_z, pert_z = z0.copy(), z0 + delta_z
        base_c, pert_c = c0.copy(), c0 + delta_z
        for _ in range(horizon):
            base_z = system.A @ base_z
            pert_z = system.A @ pert_z
            base_c = (1 - alpha) * base_c + alpha * base_z
            pert_c = (1 - alpha) * pert_c + alpha * pert_z
        true_resp = system.D @ (pert_c - base_c)
        for name, model in models.items():
            if name == "ar1":
                z = model.filter(X[: frame + 1])["latent"][-1]
                z_pert = z + np.linalg.pinv(model.observation_matrix) @ delta_obs
                base = model.decode(model.A @ z + model.b)
                pert = model.decode(model.A @ z_pert + model.b)
                pred_resp = pert - base
            elif name == "ar2":
                z_pair = model.encode(X[frame - 1 : frame + 1])
                z, z_prev = z_pair[-1], z_pair[-2]
                z_pert = z + np.linalg.pinv(model.observation_matrix) @ delta_obs
                base = model.decode(z @ model.A1.T + z_prev @ model.A2.T + model.b)
                pert = model.decode(z_pert @ model.A1.T + z_prev @ model.A2.T + model.b)
                pred_resp = pert - base
            elif name == "calcium":
                z = model.encode(X[frame - 1 : frame + 1])[-1]
                z_pert = z + np.linalg.pinv(model.observation_matrix) @ delta_obs
                drive_base = model.decode_drive(z @ model.A.T + model.b)
                drive_pert = model.decode_drive(z_pert @ model.A.T + model.b)
                c_base = (1 - model.alpha) * X[frame] + model.alpha * drive_base
                c_pert = (1 - model.alpha) * (X[frame] + delta_obs) + model.alpha * drive_pert
                pred_resp = c_pert - c_base
            else:
                raise ValueError(name)
            denom = float(np.linalg.norm(true_resp) * np.linalg.norm(pred_resp))
            cos = float(np.dot(true_resp, pred_resp) / denom) if denom > 1e-12 else float("nan")
            rows.append(
                {
                    "frame": int(frame),
                    "model": name,
                    "neuron": int(neuron),
                    "amplitude": float(amplitude),
                    "horizon": int(horizon),
                    "true_response_norm": float(np.linalg.norm(true_resp)),
                    "predicted_response_norm": float(np.linalg.norm(pred_resp)),
                    "cosine_to_true": cos,
                    "relative_response_error": float(
                        np.linalg.norm(pred_resp - true_resp) / max(1e-12, np.linalg.norm(true_resp))
                    ),
                }
            )
    return rows


def run(seed: int = 17) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, latent_rows, perturb_rows = [], [], []
    conditions = [
        ("stable_unfiltered", dict(alpha=1.0, oscillatory=False), 0.95),
        ("calcium_filtered", dict(alpha=0.25, oscillatory=False), 0.95),
        ("oscillatory_unfiltered", dict(alpha=1.0, oscillatory=True), 0.95),
        ("oscillatory_filtered", dict(alpha=0.25, oscillatory=True), 0.95),
    ]
    for name, kwargs, radius in conditions:
        system = random_stable_system(
            latent_dim=4, n_neurons=30, radius=radius, obs_sd=0.2, process_sd=0.2, seed=seed, **kwargs
        )
        X, Z = system.simulate(2800, burn_in=100)
        train_end = 1900
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[:train_end] = True
        train_starts = np.arange(2, train_end - 1)
        test_starts = np.arange(train_end + 2, len(X) - 20)
        rng = np.random.default_rng(seed)
        eval_starts = np.sort(rng.choice(test_starts, size=min(180, len(test_starts)), replace=False))
        models = _fit_models(X, train_mask, train_starts, system.latent_dim, seed=seed)
        models = {"ar1": models[0], "ar2": models[1], "calcium": models[2]}
        truth = np.stack([X[t + 1 : t + 21] for t in eval_starts])
        hist = np.stack([X[t - 1 : t + 1] for t in eval_starts])
        persistence = np.repeat(hist[:, -1:, :], 20, axis=1)
        preds = {
            "persistence": persistence,
            "ar1": _forecast_ar1(models["ar1"], hist, 20),
            "ar2": _forecast_generic(models["ar2"], hist, 20),
            "calcium": _forecast_generic(models["calcium"], hist, 20),
        }
        base = float(np.mean((persistence - truth) ** 2))
        for model_name, pred in preds.items():
            one_truth = X[eval_starts + 1]
            one_pred = pred[:, 0, :]
            rows.append(
                {
                    "condition": name,
                    "model": model_name,
                    "horizon": 20,
                    "n_starts": len(eval_starts),
                    "mse_h20": float(np.mean((pred - truth) ** 2)),
                    "persistence_mse_h20": base,
                    "skill_h20": _skill(float(np.mean((pred - truth) ** 2)), base),
                    "r2_h20": _r2(truth, pred),
                    "selected_alpha": float(getattr(models["calcium"], "alpha", np.nan))
                    if model_name == "calcium"
                    else float("nan"),
                    "realized_radius": float(
                        {
                            "ar1": np.max(np.abs(np.linalg.eigvals(models["ar1"].A))),
                            "ar2": np.max(
                                np.abs(
                                    np.linalg.eigvals(
                                        np.block(
                                            [
                                                [models["ar2"].A1, models["ar2"].A2],
                                                [np.eye(system.latent_dim), np.zeros((system.latent_dim, system.latent_dim))],
                                            ]
                                        )
                                    )
                                )
                            ),
                            "calcium": np.max(np.abs(np.linalg.eigvals(models["calcium"].A))),
                        }.get(model_name, np.nan)
                    ),
                    "one_step_mse": float(np.mean((one_pred - one_truth) ** 2)),
                }
            )

        # Latent recovery, after a linear map (rotation/sign invariant).
        true_test = Z[eval_starts]
        est = {
            "ar1": models["ar1"].encode(X[eval_starts]),
            "ar2": models["ar2"].encode(X[eval_starts]),
            "calcium": models["calcium"].pca.transform(
                _causal_calcium_transform(X, models["calcium"].alpha)[eval_starts]
            ),
        }
        for model_name, z_est in est.items():
            fwd, rev = _linear_alignment_r2(true_test, z_est)
            latent_rows.append(
                {
                    "condition": name,
                    "model": model_name,
                    "true_to_estimate_r2": fwd,
                    "estimate_to_true_r2": rev,
                    "estimated_dim": int(z_est.shape[1]),
                    "true_dim": int(system.latent_dim),
                }
            )
        perturb_frames = eval_starts[:5]
        perturb_rows.extend(
            _perturbation_rows(system, Z, models, X, perturb_frames, neuron=0, amplitude=0.5, horizon=20)
        )
    return pd.DataFrame(rows), pd.DataFrame(latent_rows), pd.DataFrame(perturb_rows)
