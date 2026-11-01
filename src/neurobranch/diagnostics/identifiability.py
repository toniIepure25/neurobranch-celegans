"""Latent-dimensionality and operator-identifiability diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from neurobranch.dynamics.context_models import StableLatentLDS


def principal_angles(Q1: np.ndarray, Q2: np.ndarray) -> dict[str, float]:
    """Principal angles between two orthonormal column spaces."""
    q1, _ = np.linalg.qr(np.asarray(Q1, dtype=np.float64))
    q2, _ = np.linalg.qr(np.asarray(Q2, dtype=np.float64))
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return {
        "mean_angle_deg": float(np.mean(angles)),
        "max_angle_deg": float(np.max(angles)),
        "mean_cos2_overlap": float(np.mean(s**2)),
        "min_cos2_overlap": float(np.min(s**2)),
    }


def pca_diagnostics(X: np.ndarray, train_mask: np.ndarray) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    train_mask = np.asarray(train_mask, dtype=bool)
    pca = PCA().fit(X[train_mask])
    ev = np.asarray(pca.explained_variance_, dtype=np.float64)
    ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    pr = float((ev.sum() ** 2) / np.sum(ev**2))
    n90 = int(np.searchsorted(np.cumsum(ratio), 0.90) + 1)
    return {
        "n_frames": int(train_mask.sum()),
        "n_neurons": int(X.shape[1]),
        "participation_ratio": pr,
        "n_components_90pct": n90,
        "top_component_fraction": float(ratio[0]),
        "top5_component_fraction": float(ratio[:5].sum()),
    }


def fit_ar1(data, *, latent_dim: int = 15, starts: np.ndarray | None = None) -> StableLatentLDS:
    if starts is None:
        from neurobranch.preprocessing import split_starts

        starts = split_starts(data, "train", history=1, horizon=1)
    return StableLatentLDS.fit(
        data.X,
        data.y_macro,
        starts,
        train_mask=data.masks["train"],
        latent_dim=latent_dim,
        ridge=1e-2,
    )


def subspace_diagnostics(data, *, latent_dim: int = 15, seed: int = 17) -> dict[str, Any]:
    train_mask = data.masks["train"]
    idx = np.flatnonzero(train_mask)
    mid = len(idx) // 2
    halves = [idx[:mid], idx[mid:]]
    qs = []
    for part in halves:
        p = PCA(n_components=latent_dim, random_state=seed).fit(data.X[part])
        qs.append(np.asarray(p.components_, dtype=np.float64).T)
    half = principal_angles(qs[0], qs[1])

    rng = np.random.default_rng(seed)
    full = PCA(n_components=latent_dim, random_state=seed).fit(data.X[train_mask])
    q_full = np.asarray(full.components_, dtype=np.float64).T
    boot = []
    for b in range(5):
        choose = rng.choice(idx, size=int(0.8 * len(idx)), replace=False)
        p = PCA(n_components=latent_dim, random_state=seed + b).fit(data.X[np.sort(choose)])
        q = np.asarray(p.components_, dtype=np.float64).T
        boot.append(principal_angles(q_full, q)["mean_angle_deg"])
    return {
        "half_subspace": half,
        "bootstrap_mean_angle_deg": float(np.mean(boot)),
        "bootstrap_max_angle_deg": float(np.max(boot)),
        "bootstrap_angles_deg": [float(x) for x in boot],
    }


def operator_diagnostics(model: StableLatentLDS) -> dict[str, Any]:
    D = np.asarray(model.observation_matrix, dtype=np.float64)
    A = np.asarray(model.A, dtype=np.float64)
    blocks = [D]
    for _ in range(model.latent_dim):
        blocks.append(blocks[-1] @ A)
    observability = np.vstack(blocks)
    s = np.linalg.svd(observability, compute_uv=False)
    eig = np.linalg.eigvals(A)
    return {
        "latent_dim": int(model.latent_dim),
        "observability_rank": int(np.linalg.matrix_rank(observability)),
        "observability_sigma_min": float(s[-1]),
        "observability_sigma_max": float(s[0]),
        "observability_condition": float(s[0] / max(1e-12, s[-1])),
        "eigenvalue_magnitudes_sorted": [float(x) for x in np.sort(np.abs(eig))[::-1]],
        "eigenvalue_angles_deg_sorted": [
            float(x) for x in np.sort(np.degrees(np.angle(eig)))[::-1]
        ],
        "realized_spectral_radius": float(np.max(np.abs(eig))),
    }


def reconstruction(data, model: StableLatentLDS) -> dict[str, float]:
    test = data.X[data.masks["test"]]
    rec = model.reconstruct(test)
    ss_res = float(np.sum((test - rec) ** 2))
    ss_tot = float(np.sum((test - test.mean(axis=0, keepdims=True)) ** 2))
    return {
        "test_reconstruction_r2": 1.0 - ss_res / ss_tot,
        "test_reconstruction_mse": float(np.mean((test - rec) ** 2)),
    }


def evaluate(data, *, latent_dim: int = 15) -> dict[str, Any]:
    model = fit_ar1(data, latent_dim=latent_dim)
    out = {
        "worm_id": int(data.worm_id),
        "pca": pca_diagnostics(data.X, data.masks["train"]),
        "subspace": subspace_diagnostics(data, latent_dim=latent_dim),
        "operator": operator_diagnostics(model),
        "reconstruction": reconstruction(data, model),
    }
    return out
