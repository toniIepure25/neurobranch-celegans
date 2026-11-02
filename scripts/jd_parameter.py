"""Stage H: parameter-uncertainty audit and a small causal bootstrap ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from neurobranch.adaptive.evaluation import evaluation_landmarks
from neurobranch.adaptive.var import RollingVAR, fit_ridge_var
from neurobranch.diagnostics.common import load_nostim
from neurobranch.joint_dynamics.evaluation import autocorrelation_error, neural_covariance_error
from neurobranch.joint_dynamics.models import block_bootstrap_starts
from neurobranch.joint_dynamics.residuals import lead1_innovations
from neurobranch.probabilistic.metrics import coverage_summary, crps_ensemble, energy_score
from neurobranch.probabilistic.residual import adaptive_prequential_residuals
from neurobranch.uncertainty_calibration.trajectory import trajectory_energy_score

OUT = Path("reports/simulator/joint_dynamics")
DRAWS = 5
MEMBERS_PER_DRAW = 20
BLOCK = 25


def _bootstrap_models(X: np.ndarray, t: int, *, window: int = 500, draws: int = DRAWS) -> list:
    starts = np.arange(max(1, int(t) - window + 1), int(t), dtype=np.int64)
    if len(starts) < 100:
        return []
    rng = np.random.default_rng(4400 + int(t))
    out = []
    for _ in range(draws):
        sampled = block_bootstrap_starts(starts, block=BLOCK, rng=rng)
        fit = fit_ridge_var(X, sampled, ridge=1.0, spectral_cap=0.995)
        if fit is not None:
            A, b, record = fit
            out.append((A, b, record))
    return out


def synthetic_audit() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    A = np.array([[0.72, 0.08], [-0.10, 0.62]])
    Q = np.diag([0.20, 0.12])
    n = 1000
    X = np.zeros((n, 2))
    for t in range(1, n):
        X[t] = A @ X[t - 1] + rng.multivariate_normal(np.zeros(2), Q)
    train = X[:700]
    starts = np.arange(1, 699)
    fit = fit_ridge_var(train, starts, ridge=1.0, spectral_cap=0.995)
    Ahat, bhat, _ = fit
    draws = []
    for d in range(20):
        sampled = block_bootstrap_starts(starts, block=25, rng=np.random.default_rng(800 + d))
        f = fit_ridge_var(train, sampled, ridge=1.0, spectral_cap=0.995)
        if f is not None:
            draws.append((f[0], f[1]))
    x0 = X[700]
    horizon = 20
    n_truth = 200
    truth_paths = np.empty((n_truth, horizon, 2))
    for r in range(n_truth):
        x = x0.copy()
        for step in range(horizon):
            x = A @ x + rng.multivariate_normal(np.zeros(2), Q)
            truth_paths[r, step] = x
    rows = []
    for name, use_draws in (("innovation_only", False), ("parameter_ensemble", True), ("oracle", False)):
        members = 200
        ens = np.empty((members, horizon, 2))
        for m in range(members):
            if use_draws:
                Am, bm = draws[m % len(draws)]
            elif name == "oracle":
                Am, bm = A, np.zeros(2)
            else:
                Am, bm = Ahat, bhat
            x = x0.copy()
            for step in range(horizon):
                x = Am @ x + bm + rng.multivariate_normal(np.zeros(2), Q)
                ens[m, step] = x
        coverages = [coverage_summary(ens[:, -1], truth_paths[r, -1]) for r in range(n_truth)]
        cov = {f"coverage_{level}": float(np.mean([c[f"coverage_{level}"] for c in coverages])) for level in (50, 80, 95)}
        widths = float(np.mean([c["width_95"] for c in coverages]))
        energies = float(np.mean([energy_score(ens[:, -1], truth_paths[r, -1]) for r in range(n_truth)]))
        rows.append(
            {
                "model": name,
                "coverage_50": cov["coverage_50"],
                "coverage_80": cov["coverage_80"],
                "coverage_95": cov["coverage_95"],
                "width_95": widths,
                "energy_h20": energies,
                "ensemble_std": float(ens[:, -1].std(0).mean()),
                "n_parameter_draws": len(draws) if use_draws else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    synth = synthetic_audit()
    synth.to_csv(OUT / "parameter_synthetic_audit.csv", index=False)
    _, data, _ = load_nostim()
    rows, draw_rows = [], []
    for d in data:
        sid = int(d.worm_id)
        pca = PCA(n_components=10, random_state=17).fit(d.X[d.masks["train"]])
        series = adaptive_prequential_residuals(d, window=500, block=10)
        for t in evaluation_landmarks(d, horizon_max=100, stride=10):
            t = int(t)
            base = RollingVAR(window=500).fit_at(d.X, t)
            if base is None:
                continue
            models = _bootstrap_models(d.X, t)
            if len(models) < 2:
                continue
            sample = lead1_innovations(series, t=t)
            scale = np.sqrt(np.maximum(sample.values.var(axis=0, ddof=1), 1e-10))
            ens = np.empty((DRAWS * MEMBERS_PER_DRAW, 100, d.X.shape[1]))
            for k, (A, b, _) in enumerate(models):
                eps = np.random.default_rng(7000 + sid * 10000 + t + 31 * k).standard_normal(
                    (MEMBERS_PER_DRAW, 100, d.X.shape[1])
                ) * scale
                x = np.repeat(d.X[t][None, :], MEMBERS_PER_DRAW, axis=0)
                for step in range(100):
                    x = x @ A.T + b + eps[:, step]
                    ens[k * MEMBERS_PER_DRAW : (k + 1) * MEMBERS_PER_DRAW, step] = x
            truth = d.X[t + 1 : t + 101]
            truth_p = pca.transform(truth)
            ens_p = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(ens.shape[0], 100, -1)
            row = {"model": "H_parameter_ensemble", "worm_id": sid, "landmark": t}
            for h in (1, 5, 10, 20, 50, 100):
                c = coverage_summary(ens_p[:, h - 1], truth_p[h - 1])
                row[f"energy_h{h}"] = energy_score(ens_p[:, h - 1], truth_p[h - 1])
                row[f"crps_h{h}"] = float(crps_ensemble(ens_p[:, h - 1], truth_p[h - 1]).mean())
                for level in (50, 80, 95):
                    row[f"coverage_h{h}_{level}"] = c[f"coverage_{level}"]
            row["trajectory_energy_h20"] = trajectory_energy_score(ens_p[:, :20], truth_p[:20])
            row["trajectory_energy_h100"] = trajectory_energy_score(ens_p, truth_p)
            row["neural_cov_error"] = neural_covariance_error(ens_p[:, :20], truth_p[:20])
            row["autocorr_error"] = autocorrelation_error(ens_p[:, :20], truth_p[:20])
            rows.append(row)
            draw_rows.append(
                {
                    "worm_id": sid,
                    "landmark": t,
                    "n_draws": len(models),
                    "a_spread": float(np.std([np.linalg.norm(m[0]) for m in models])),
                    "b_spread": float(np.std([np.linalg.norm(m[1]) for m in models])),
                }
            )
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "parameter_scores.csv", index=False)
    pd.DataFrame(draw_rows).to_csv(OUT / "parameter_draws.csv", index=False)
    summary = scores.agg(
        {
            "energy_h20": "mean",
            "crps_h20": "mean",
            "trajectory_energy_h20": "mean",
            "trajectory_energy_h100": "mean",
            "coverage_h20_50": "mean",
            "coverage_h20_80": "mean",
            "coverage_h20_95": "mean",
            "neural_cov_error": "mean",
            "autocorr_error": "mean",
        }
    )
    summary.to_frame().T.to_csv(OUT / "parameter_summary.csv", index=False)
    (OUT / "parameter_decision.json").write_text(
        json.dumps(
            {
                "method": "moving-block bootstrap of causal transition pairs, 5 draws x 20 members",
                "block_length": BLOCK,
                "window": 500,
                "draws": DRAWS,
                "process_covariance": "diagonal from lead-1 residuals, held fixed across parameter draws",
                "initial_state_uncertainty": "none; observed initial state used",
                "observation_uncertainty": "not separately identifiable",
                "test_refit": False,
                "accepted": False,
                "reason": "real-data parameter ensemble worsens energy, trajectory and coverage relative to the diagonal population model",
                "synthetic": synth.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(synth.to_string())
    print(summary.to_string())


if __name__ == "__main__":
    main()
