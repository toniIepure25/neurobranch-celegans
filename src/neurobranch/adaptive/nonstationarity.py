"""Sliding-window nonstationarity diagnostics with matched nulls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from sklearn.decomposition import PCA

from neurobranch.adaptive.var import fit_ridge_var, spectral_radius


def principal_angles_deg(Q1: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    q1, _ = np.linalg.qr(np.asarray(Q1, dtype=np.float64))
    q2, _ = np.linalg.qr(np.asarray(Q2, dtype=np.float64))
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def subspace_and_spectrum(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    pca = PCA(n_components=min(k, X.shape[0], X.shape[1]), random_state=17).fit(X)
    ev = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    pr = float((ev.sum() ** 2) / np.sum(ev**2)) if np.sum(ev**2) > 0 else float("nan")
    return np.asarray(pca.components_.T, dtype=np.float64), ev, pr


def _local_ar(X: np.ndarray) -> tuple[np.ndarray, float]:
    fitted = fit_ridge_var(X, np.arange(1, len(X) - 1), ridge=1.0, spectral_cap=None)
    if fitted is None:
        return np.full((X.shape[1], X.shape[1]), np.nan), float("nan")
    A, _b, record = fitted
    return A, spectral_radius(A)


def window_stats(
    X: np.ndarray,
    *,
    reference_mask: np.ndarray,
    window: int = 256,
    step: int = 64,
    k: int = 10,
) -> pd.DataFrame:
    X = np.asarray(X, dtype=np.float64)
    ref = X[np.asarray(reference_mask, dtype=bool)]
    Qref, _ev, pr_ref = subspace_and_spectrum(ref, k)
    A_ref, _rho_ref = _local_ar(ref)
    cov_ref = np.cov(ref, rowvar=False)
    std_ref = ref.std(axis=0)
    med_ref = np.median(ref, axis=0)
    rows = []
    prev_q: np.ndarray | None = None
    for start in range(0, max(0, len(X) - window + 1), step):
        seg = X[start : start + window]
        q, _ev, pr = subspace_and_spectrum(seg, k)
        angles_ref = principal_angles_deg(q, Qref)
        if prev_q is not None:
            angles_adj = principal_angles_deg(q, prev_q)
            adj_mean = float(np.mean(angles_adj))
        else:
            adj_mean = float("nan")
        prev_q = q
        cov = np.cov(seg, rowvar=False)
        cov_rel = float(np.linalg.norm(cov - cov_ref) / max(1e-12, np.linalg.norm(cov_ref)))
        std = seg.std(axis=0)
        amp = float(np.mean(np.abs(np.log((std + 1e-8) / (std_ref + 1e-8)))))
        occ = float(np.mean(np.abs(np.median(seg, axis=0) - med_ref) / (std_ref + 1e-8)))
        A_loc, rho = _local_ar(seg)
        if np.isfinite(A_loc).all() and np.isfinite(A_ref).all():
            ar_drift = float(np.linalg.norm(A_loc - A_ref) / max(1e-12, np.linalg.norm(A_ref)))
        else:
            ar_drift = float("nan")
        rows.append(
            {
                "window_start": int(start),
                "window_stop": int(start + window),
                "window_center": float(start + window / 2),
                "reference_angle_mean_deg": float(np.mean(angles_ref)),
                "reference_angle_max_deg": float(np.max(angles_ref)),
                "adjacent_angle_mean_deg": adj_mean,
                "participation_ratio": pr,
                "reference_participation_ratio": pr_ref,
                "covariance_relative_change": cov_rel,
                "amplitude_shift": amp,
                "occupancy_shift": occ,
                "local_ar_relative_change": ar_drift,
                "local_spectral_radius": rho,
            }
        )
    return pd.DataFrame(rows)


def phase_surrogate(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multivariate phase randomization preserving power and cross-spectrum."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    spec = np.fft.rfft(X, axis=0)
    phases = np.exp(2j * np.pi * rng.random(len(spec)))
    phases[0] = 1.0
    if n % 2 == 0:
        phases[-1] = 1.0
    out = np.fft.irfft(spec * phases[:, None], n=n, axis=0)
    return out


def stationary_null(
    X: np.ndarray,
    reference_mask: np.ndarray,
    *,
    window: int = 256,
    step: int = 64,
    k: int = 10,
    n_surrogates: int = 8,
    seed: int = 17,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(n_surrogates)):
        surrogate = phase_surrogate(X, rng)
        stats = window_stats(
            surrogate, reference_mask=reference_mask, window=window, step=step, k=k
        )
        stats["surrogate"] = i
        rows.append(stats)
    return pd.concat(rows, ignore_index=True)


def synthetic_observed(
    *,
    n_frames: int = 3000,
    latent_dim: int = 10,
    n_neurons: int = 40,
    regime: bool = False,
    seed: int = 17,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = latent_dim
    # Stable block-diagonal rotations with slow modes.
    blocks = []
    for _ in range(k // 2):
        theta = rng.uniform(0.05, 0.35)
        rho = rng.uniform(0.85, 0.97)
        blocks.append(rho * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]))
    A1 = block_diag(*blocks) if blocks else np.array([[0.9]])
    if A1.shape[0] < k:
        A1 = np.pad(A1, ((0, k - A1.shape[0]), (0, k - A1.shape[0])))
    Q, _ = np.linalg.qr(rng.normal(size=(k, k)))
    A2 = Q @ A1 @ Q.T
    D = rng.normal(size=(k, n_neurons)) / np.sqrt(k)
    D2 = rng.normal(size=(k, n_neurons)) / np.sqrt(k)
    z = rng.normal(size=k)
    obs = np.empty((n_frames, n_neurons), dtype=np.float64)
    for t in range(n_frames):
        if regime and t >= n_frames // 2:
            z = A2 @ z + rng.normal(scale=0.15, size=k)
            obs[t] = D2.T @ z + rng.normal(scale=0.2, size=n_neurons)
        else:
            z = A1 @ z + rng.normal(scale=0.15, size=k)
            obs[t] = D.T @ z + rng.normal(scale=0.2, size=n_neurons)
    return obs


def summarize_windows(
    windows: pd.DataFrame, null: pd.DataFrame, *, late_fraction: float = 0.6
) -> dict[str, Any]:
    cutoff = windows.window_start.quantile(late_fraction)
    late = windows[windows.window_start >= cutoff]
    early = windows[windows.window_start < cutoff]
    null_mean = null.reference_angle_mean_deg.quantile(0.95)
    null_max = null.reference_angle_max_deg.quantile(0.95)
    return {
        "n_windows": int(len(windows)),
        "early_reference_angle_mean_deg": float(early.reference_angle_mean_deg.mean()),
        "late_reference_angle_mean_deg": float(late.reference_angle_mean_deg.mean()),
        "max_reference_angle_mean_deg": float(windows.reference_angle_mean_deg.max()),
        "late_covariance_relative_change": float(late.covariance_relative_change.mean()),
        "late_amplitude_shift": float(late.amplitude_shift.mean()),
        "late_occupancy_shift": float(late.occupancy_shift.mean()),
        "late_local_ar_relative_change": float(late.local_ar_relative_change.mean()),
        "surrogate_angle_mean_95pct": float(null_mean),
        "surrogate_angle_max_95pct": float(null_max),
        "observed_exceeds_surrogate_mean": bool(
            late.reference_angle_mean_deg.mean() > null_mean
        ),
    }
