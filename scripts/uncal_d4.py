"""D4 diagnostic: validation-fitted global variance scaling.

This is a calibration baseline, not a mechanistic model. The scaling factor is
fitted only on the validation split and applied unchanged to test landmarks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from neurobranch.adaptive.evaluation import evaluation_landmarks
from neurobranch.diagnostics.common import load_nostim
from neurobranch.preprocessing import split_starts
from neurobranch.probabilistic.baselines import IndependentARModel
from neurobranch.probabilistic.metrics import coverage_summary, crps_ensemble, energy_score
from neurobranch.probabilistic.residual import independent_ar_residuals
from neurobranch.probabilistic.stochastic import Innovation, ensemble_rollout
from neurobranch.uncertainty_calibration.models import diagonal_scale
from neurobranch.uncertainty_calibration.trajectory import trajectory_energy_score

OUT = Path("reports/simulator/uncertainty_calibration")
HORIZONS = (1, 5, 10, 20, 50, 100)


def _pool(series, t: int) -> np.ndarray:
    mask = (series.targets <= int(t)) & (series.lead == 1)
    return series.values[mask][-300:]


def main(n_members: int = 100) -> None:
    cfg, data, _ = load_nostim()
    val_rows, score_rows, cov_rows, traj_rows = [], [], [], []
    factors = {}
    for d in data:
        sid = int(d.worm_id)
        series = independent_ar_residuals(d, window=200, block=10)
        val_landmarks = split_starts(d, "val", history=2, horizon=20)[::5]
        standard = []
        for t in val_landmarks:
            t = int(t)
            fitted = IndependentARModel(window=200).fit_at(d.X, t)
            pool = _pool(series, t)
            if fitted is None or len(pool) < 25:
                continue
            scale = diagonal_scale(pool)
            ens = ensemble_rollout(
                "independent_ar", fitted, d.X[t], 20, n_members,
                Innovation("gauss_diag", scale=scale, metadata={}),
                np.random.default_rng(9000 + sid * 10000 + t),
            )
            err = (ens[:, 19, :].mean(0) - d.X[t + 20]) ** 2
            var = ens[:, 19, :].var(0, ddof=1)
            standard.extend((err / np.maximum(var, 1e-8)).tolist())
        factor = float(np.clip(np.mean(standard), 0.25, 4.0)) if standard else 1.0
        factors[sid] = factor
        val_rows.append({"worm_id": sid, "n_val_landmarks": len(val_landmarks), "variance_factor": factor})
        print("worm", sid, "factor", factor, "n", len(val_landmarks))

    for d in data:
        sid = int(d.worm_id)
        series = independent_ar_residuals(d, window=200, block=10)
        train = d.X[d.masks["train"]]
        pca = PCA(n_components=10, random_state=17).fit(train)
        factor = factors[sid]
        for t in evaluation_landmarks(d, horizon_max=100, stride=10):
            t = int(t)
            fitted = IndependentARModel(window=200).fit_at(d.X, t)
            pool = _pool(series, t)
            if fitted is None or len(pool) < 25:
                continue
            scale = diagonal_scale(pool) * np.sqrt(factor)
            ens = ensemble_rollout(
                "independent_ar", fitted, d.X[t], 100, n_members,
                Innovation("gauss_diag", scale=scale, metadata={}),
                np.random.default_rng(7000 + sid * 10000 + t),
            )
            truth = d.X[t + 1 : t + 101]
            ens_proj = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(n_members, 100, -1)
            truth_proj = pca.transform(truth)
            cov = coverage_summary(ens[:, 19, :], truth[19])
            crps = crps_ensemble(ens[:, 19, :], truth[19])
            score_rows.append(
                {
                    "worm_id": sid,
                    "landmark": t,
                    "model": "D4_scaled_diag",
                    "crps_h20": float(crps.mean()),
                    "coverage_50": cov["coverage_50"],
                    "coverage_80": cov["coverage_80"],
                    "coverage_95": cov["coverage_95"],
                    "width_95": cov["width_95"],
                    "energy_h20": energy_score(ens_proj[:, 19, :], truth_proj[19]),
                    "trajectory_energy_h20": trajectory_energy_score(ens_proj[:, :20, :], truth_proj[:20]),
                }
            )
            for h in HORIZONS:
                ch = coverage_summary(ens[:, h - 1, :], truth[h - 1])
                cov_rows.append(
                    {
                        "worm_id": sid,
                        "landmark": t,
                        "model": "D4_scaled_diag",
                        "horizon": h,
                        "coverage_50": ch["coverage_50"],
                        "coverage_80": ch["coverage_80"],
                        "coverage_95": ch["coverage_95"],
                        "width_95": ch["width_95"],
                    }
                )
            traj_rows.append(
                {
                    "worm_id": sid,
                    "landmark": t,
                    "model": "D4_scaled_diag",
                    "trajectory_energy_h20": score_rows[-1]["trajectory_energy_h20"],
                }
            )
    pd.DataFrame(val_rows).to_csv(OUT / "d4_validation.csv", index=False)
    pd.DataFrame(score_rows).to_csv(OUT / "d4_scores.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(OUT / "d4_coverage.csv", index=False)
    pd.DataFrame(traj_rows).to_csv(OUT / "d4_trajectory.csv", index=False)
    summary = {
        "factors": factors,
        "mean_factor": float(np.mean(list(factors.values()))),
        "note": "Validation-fitted global variance scaling; diagnostic only.",
    }
    (OUT / "d4_summary.json").write_text(json.dumps(summary, indent=2))
    print("D4 complete", len(score_rows))


if __name__ == "__main__":
    main()
