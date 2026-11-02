"""Known-ground-truth audits for coupling recovery and common-input confusion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.kato_interactions.models import fit_var
from neurobranch.kato_interactions.sensitivity import response_cosine


def _system(kind: str, *, n: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    A = np.eye(n) * 0.75
    if kind in {"coupled", "coupled_common"}:
        for j in range(n):
            A[j, (j + 1) % n] = 0.12
            A[j, (j - 1) % n] = 0.06
    loading = None
    if kind in {"common", "coupled_common"}:
        loading = rng.normal(0.5, 0.1, size=n)
    return A, loading


def simulate(
    kind: str,
    *,
    n: int = 8,
    n_frames: int = 2400,
    seed: int = 0,
    burn: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    A, loading = _system(kind, n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    x = np.zeros(n, dtype=np.float64)
    s = 0.0
    rows = []
    for t in range(int(n_frames) + int(burn)):
        if loading is not None:
            s = 0.9 * s + rng.normal(scale=0.5)
        x = A @ x + rng.normal(scale=0.35, size=n)
        if loading is not None:
            x = x + loading * s
        if t >= burn:
            rows.append(x.copy())
    return np.asarray(rows), A, loading


def evaluate_system(kind: str, *, seed: int = 0, n_frames: int = 2400, n: int = 8) -> dict[str, float]:
    X, A_true, _ = simulate(kind, n=n, n_frames=n_frames, seed=seed)
    starts = np.arange(0, len(X) - 1, dtype=np.int64)
    full = fit_var(X, starts, ridge=1.0, spectral_cap=None)
    diag = fit_var(X, starts, ridge=1.0, spectral_cap=None, diagonal=True)
    if full is None or diag is None:
        raise RuntimeError("synthetic fit failed")
    test = np.arange(len(X) - 250, len(X) - 20, dtype=np.int64)
    err_full, err_diag = [], []
    for t in test:
        truth = X[int(t) + 20]
        err_full.append(float(np.mean((full.forecast_mean(X[int(t)], 20)[-1] - truth) ** 2)))
        err_diag.append(float(np.mean((diag.forecast_mean(X[int(t)], 20)[-1] - truth) ** 2)))
    true_off = A_true - np.diag(np.diag(A_true))
    fit_off = full.A - np.diag(np.diag(full.A))
    mask = ~np.eye(n, dtype=bool)
    corr = float(np.corrcoef(true_off[mask], fit_off[mask])[0, 1]) if np.std(true_off) > 0 else float("nan")
    impulse_true = np.linalg.matrix_power(A_true, 20)[:, 0]
    impulse_fit = np.linalg.matrix_power(full.A, 20)[:, 0]
    return {
        "scenario": kind,
        "seed": int(seed),
        "true_offdiag_norm": float(np.linalg.norm(true_off)),
        "fit_offdiag_norm": float(np.linalg.norm(fit_off)),
        "offdiag_recovery_corr": corr,
        "full_mse_h20": float(np.mean(err_full)),
        "diag_mse_h20": float(np.mean(err_diag)),
        "mse_improvement_diag_minus_full": float(np.mean(err_diag) - np.mean(err_full)),
        "forecast_ratio_full_over_diag": float(np.mean(err_full) / max(np.mean(err_diag), 1e-12)),
        "impulse_cosine_truth_fit": response_cosine(impulse_true, impulse_fit),
        "fit_spectral_radius": full.spectral_radius,
        "true_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A_true)))),
    }


def audit(seeds: tuple[int, ...] = (11, 23, 37)) -> pd.DataFrame:
    rows = []
    for kind in ("independent", "coupled", "common", "coupled_common"):
        for seed in seeds:
            rows.append(evaluate_system(kind, seed=seed))
    return pd.DataFrame(rows)
