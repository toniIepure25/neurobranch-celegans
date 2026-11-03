"""Independent reference calculation for B3 and B5 at selected landmarks.

The project functions are not used for the mean models, PCA, rollouts, or energy
score.  Only the historical J5 innovation sampler is reused, because the exact
random innovation path must match the frozen comparison.  This is a code/math
cross-check, not a new experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import jd_evaluate as JD  # noqa: E402

from neurobranch.diagnostics.common import load_nostim  # noqa: E402

OUT = ROOT / "reports" / "master_review"
WINDOW = 500
RIDGE = 1.0
CAP = 0.995
K = 10
MEMBERS = 100
HORIZON = 100
CASES = ((0, 2690), (2, 2624), (4, 2592))


def ridge_var(X: np.ndarray, starts: np.ndarray, ridge: float = RIDGE, cap: float | None = CAP):
    starts = starts[(starts >= 0) & (starts + 1 < len(X))]
    design = X[starts].astype(np.float64)
    target = X[starts + 1].astype(np.float64)
    mu_x = design.mean(axis=0)
    mu_y = target.mean(axis=0)
    sd = design.std(axis=0)
    sd[sd < 1e-8] = 1.0
    xs = (design - mu_x) / sd
    coef = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ (target - mu_y))
    A = (coef / sd[:, None]).T
    b = mu_y - (mu_x / sd) @ coef
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if cap is not None and radius > cap:
        A = A * (cap / radius)
        radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    return A, b, {"raw_radius": float(np.max(np.abs(np.linalg.eigvals(A)))) if cap is None else radius, "radius": radius}


def pca_reference(train: np.ndarray, k: int = K):
    mean = train.mean(axis=0)
    centered = train - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:k]
    # Deterministic sign convention; predictions are invariant to component signs.
    for j in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[j])))
        if components[j, pivot] < 0:
            components[j] *= -1
    return mean, components


def scores(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (X.astype(np.float64) - mean) @ components.T


def energy(ensemble: np.ndarray, observation: np.ndarray) -> float:
    ens = np.asarray(ensemble, dtype=np.float64)
    obs = np.asarray(observation, dtype=np.float64).ravel()
    term1 = np.linalg.norm(ens - obs, axis=1).mean()
    m = len(ens)
    diff = ens[:, None, :] - ens[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    term2 = 0.5 * dist[np.triu_indices(m, k=1)].mean()
    return float(term1 - term2)


def b3_reference(X: np.ndarray, t: int, eps: np.ndarray):
    starts = np.arange(max(1, t - WINDOW + 1), t, dtype=np.int64)
    A, b, diag = ridge_var(X, starts)
    x = np.repeat(X[t].astype(np.float64)[None, :], eps.shape[0], axis=0)
    out = np.empty((eps.shape[0], eps.shape[1], X.shape[1]), dtype=np.float64)
    for step in range(eps.shape[1]):
        x = x @ A.T + b + eps[:, step]
        out[:, step] = x
    return out, diag


def b5_reference(X: np.ndarray, mean: np.ndarray, components: np.ndarray, t: int, eps: np.ndarray):
    F = scores(X, mean, components)
    starts = np.arange(max(1, t - WINDOW + 1), t, dtype=np.int64)
    design = X[starts].astype(np.float64)
    target = X[starts + 1].astype(np.float64)
    f_design = F[starts, :K]
    factor_A, factor_b, factor_diag = ridge_var(F[:, :K], starts)

    mu_x = design.mean(axis=0)
    sd_x = design.std(axis=0)
    sd_x[sd_x < 1e-8] = 1.0
    mu_f = f_design.mean(axis=0)
    sd_f = f_design.std(axis=0)
    sd_f[sd_f < 1e-8] = 1.0
    xs = (design - mu_x) / sd_x
    fs = (f_design - mu_f) / sd_f
    coef = np.empty((design.shape[1], 1 + K), dtype=np.float64)
    b = np.empty(design.shape[1], dtype=np.float64)
    for j in range(design.shape[1]):
        design2 = np.column_stack([xs[:, j], fs])
        y = target[:, j] - target[:, j].mean()
        coef[j] = np.linalg.solve(design2.T @ design2 + RIDGE * np.eye(1 + K), design2.T @ y)
        b[j] = target[:, j].mean()

    x = np.repeat(X[t].astype(np.float64)[None, :], eps.shape[0], axis=0)
    f = np.repeat(F[t, :K][None, :], eps.shape[0], axis=0)
    out = np.empty((eps.shape[0], eps.shape[1], X.shape[1]), dtype=np.float64)
    for step in range(eps.shape[1]):
        xs_t = (x - mu_x) / sd_x
        fs_t = (f - mu_f) / sd_f
        x = b + coef[:, 0] * xs_t + fs_t @ coef[:, 1:].T + eps[:, step]
        f = f @ factor_A.T + factor_b
        out[:, step] = x
    return out, {"factor_radius": factor_diag["radius"], "n_parameters": int(coef.size + factor_A.size + factor_b.size + len(b))}


def main() -> None:
    cfg, data, recordings = load_nostim()
    frozen = pd.read_csv(ROOT / "reports" / "simulator" / "kato_common_state" / "model_per_landmark.csv")
    results = []
    for sid, t in CASES:
        d = data[sid]
        truth16 = d.X[t + 1 : t + HORIZON + 1]
        _, series, _ = JD.prepare(d)
        spec = JD.Spec("J5_population_lead1_boot", "var", "population", "block", 10)
        eps = JD.make_innovation(spec, series["population"], t, {}).sample(
            MEMBERS, HORIZON, np.random.default_rng(7000 + sid * 10000 + t)
        )
        mean, components = pca_reference(d.X[d.masks["train"]], K)
        # The saved run uses sklearn PCA; compare transformed training variance.
        train_scores = scores(d.X[d.masks["train"]], mean, components)
        b3_ens, b3_diag = b3_reference(d.X, t, eps)
        b5_ens, b5_diag = b5_reference(d.X, mean, components, t, eps)

        def project(arr, pca_mean=mean, pca_components=components):
            return (arr.astype(np.float64) - pca_mean) @ pca_components.T

        truth_proj = project(truth16)
        b3_proj = project(b3_ens)
        b5_proj = project(b5_ens)
        b3_energy = energy(b3_proj[:, 19], truth_proj[19])
        b5_energy = energy(b5_proj[:, 19], truth_proj[19])
        b3_mse = float(np.mean((b3_proj[:, 19].mean(axis=0) - truth_proj[19]) ** 2))
        b5_mse = float(np.mean((b5_proj[:, 19].mean(axis=0) - truth_proj[19]) ** 2))

        frozen_rows = frozen[(frozen["session"] == sid) & (frozen["landmark"] == t)]
        frozen_b3 = frozen_rows[frozen_rows["model"] == "B3_population"].iloc[0]
        frozen_b5 = frozen_rows[frozen_rows["model"] == "B5_common_k10"].iloc[0]
        results.append(
            {
                "session": sid,
                "landmark": t,
                "reference": {
                    "B3_energy_h20": b3_energy,
                    "B5_energy_h20": b5_energy,
                    "B3_mse_h20": b3_mse,
                    "B5_mse_h20": b5_mse,
                    "B3_parameters": int(d.X.shape[1] * (d.X.shape[1] + 1)),
                    "B5_parameters": b5_diag["n_parameters"],
                    "B3_spectral_radius": b3_diag["radius"],
                    "B5_factor_radius": b5_diag["factor_radius"],
                    "train_pc1_variance_reference": float(train_scores[:, 0].var(ddof=1)),
                },
                "frozen": {
                    "B3_energy_h20": float(frozen_b3["pca10_energy_h20"]),
                    "B5_energy_h20": float(frozen_b5["pca10_energy_h20"]),
                    "B3_mse_h20": float(frozen_b3["pca10_mse_mean_h20"]),
                    "B5_mse_h20": float(frozen_b5["pca10_mse_mean_h20"]),
                },
                "absolute_difference": {
                    "B3_energy": abs(b3_energy - float(frozen_b3["pca10_energy_h20"])),
                    "B5_energy": abs(b5_energy - float(frozen_b5["pca10_energy_h20"])),
                    "B3_mse": abs(b3_mse - float(frozen_b3["pca10_mse_mean_h20"])),
                    "B5_mse": abs(b5_mse - float(frozen_b5["pca10_mse_mean_h20"])),
                },
            }
        )
    payload = {
        "status": "PASS" if all(max(r["absolute_difference"].values()) < 2e-5 for r in results) else "FAIL",
        "tolerance": 2e-5,
        "reason_for_tolerance": "PCA sign/rounding and float32-to-float64 reference path; not a reproduction tolerance.",
        "cases": results,
        "note": "Only the J5 innovation sampler is reused to keep the common random path fixed; models, PCA, rollouts and energy are independent reference implementations.",
    }
    (OUT / "independent_model_check.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
