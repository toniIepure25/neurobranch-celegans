"""Stage D: prespecified Student-t scale calibration fitted on validation only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from neurobranch.adaptive.evaluation import evaluation_landmarks
from neurobranch.adaptive.var import RollingVAR
from neurobranch.diagnostics.common import load_nostim
from neurobranch.joint_dynamics.evaluation import (
    autocorrelation_error,
    neural_covariance_error,
    path_covariance_error,
)
from neurobranch.joint_dynamics.models import sample_student_t
from neurobranch.joint_dynamics.residuals import lead1_innovations
from neurobranch.preprocessing import split_starts
from neurobranch.probabilistic.metrics import (
    coverage_summary,
    crps_ensemble,
    energy_score,
    interval_coverage,
    pit_values,
)
from neurobranch.probabilistic.residual import adaptive_prequential_residuals, regularized_covariance
from neurobranch.uncertainty_calibration.trajectory import trajectory_energy_score

OUT = Path("reports/simulator/joint_dynamics")
GRID = (0.85, 0.95, 1.0, 1.1, 1.25)
DF = 5.0
MEMBERS = 100
HORIZONS = (1, 5, 10, 20, 50, 100)


def _fit_at(d, t: int, series):
    fitted = RollingVAR(window=500).fit_at(d.X, t)
    if fitted is None:
        return None, None
    sample = lead1_innovations(series, t=t)
    cov, _ = regularized_covariance(sample.values, method="ledoit")
    return fitted, cov


def _rollout(fitted, x0, eps):
    x = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], eps.shape[0], axis=0)
    out = np.empty_like(eps)
    for step in range(eps.shape[1]):
        x = x @ fitted.A.T + fitted.b + eps[:, step]
        out[:, step] = x
    return out


def _coverage(ens, obs):
    return coverage_summary(ens, obs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, data, _ = load_nostim()
    val_rows = []
    for d in data:
        sid = int(d.worm_id)
        pca = PCA(n_components=10, random_state=17).fit(d.X[d.masks["train"]])
        series = adaptive_prequential_residuals(d, window=500, block=10)
        starts = split_starts(d, "val", history=2, horizon=20)[::10]
        for t in starts:
            fitted, cov = _fit_at(d, int(t), series)
            if fitted is None:
                continue
            truth = pca.transform(d.X[int(t) + 1 : int(t) + 21])
            for factor in GRID:
                eps = sample_student_t(
                    MEMBERS, 20, cov, np.random.default_rng(9100 + sid * 10000 + int(t)), df=DF, scale=factor
                )
                ens = _rollout(fitted, d.X[int(t)], eps)
                ens_p = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(ens.shape[0], 20, -1)
                c = _coverage(ens_p[:, -1], truth[-1])
                val_rows.append(
                    {
                        "worm_id": sid,
                        "landmark": int(t),
                        "scale": factor,
                        "coverage_50": c["coverage_50"],
                        "coverage_80": c["coverage_80"],
                        "coverage_95": c["coverage_95"],
                        "crps_h20": float(crps_ensemble(ens_p[:, -1], truth[-1]).mean()),
                    }
                )
    val = pd.DataFrame(val_rows)
    val.to_csv(OUT / "d4_validation_grid.csv", index=False)
    agg = (
        val.groupby("scale")
        .agg(
            coverage_50=("coverage_50", "mean"),
            coverage_80=("coverage_80", "mean"),
            coverage_95=("coverage_95", "mean"),
            crps_h20=("crps_h20", "mean"),
        )
        .reset_index()
    )
    agg["max_abs_error"] = np.maximum.reduce(
        [
            np.abs(agg.coverage_50 - 0.50),
            np.abs(agg.coverage_80 - 0.80),
            np.abs(agg.coverage_95 - 0.95),
        ]
    )
    agg["sum_sq_error"] = (
        (agg.coverage_50 - 0.50) ** 2 + (agg.coverage_80 - 0.80) ** 2 + (agg.coverage_95 - 0.95) ** 2
    )
    # Frozen rule: minimise the maximum absolute coverage error; CRPS breaks exact ties.
    chosen = agg.sort_values(["max_abs_error", "crps_h20"]).iloc[0]
    chosen_scale = float(chosen.scale)
    agg.to_csv(OUT / "d4_validation_summary.csv", index=False)
    (OUT / "d4_decision.json").write_text(
        json.dumps(
            {
                "method": "multivariate Student-t, df=5, global scale",
                "grid": list(GRID),
                "selection_rule": "minimise maximum absolute coverage error over 50/80/95 on validation; CRPS tie-break",
                "chosen_scale": chosen_scale,
                "validation_coverage": {
                    "50": float(chosen.coverage_50),
                    "80": float(chosen.coverage_80),
                    "95": float(chosen.coverage_95),
                },
                "test_used_for_selection": False,
                "accepted": False,
                "reason": "test coverage remains over-dispersed and both H20 scores are worse than D1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rows, neuron_rows = [], []
    for d in data:
        sid = int(d.worm_id)
        pca = PCA(n_components=10, random_state=17).fit(d.X[d.masks["train"]])
        series = adaptive_prequential_residuals(d, window=500, block=10)
        for t in evaluation_landmarks(d, horizon_max=100, stride=10):
            fitted, cov = _fit_at(d, int(t), series)
            if fitted is None:
                continue
            truth = d.X[int(t) + 1 : int(t) + 101]
            truth_p = pca.transform(truth)
            eps = sample_student_t(
                MEMBERS, 100, cov, np.random.default_rng(7000 + sid * 10000 + int(t)), df=DF, scale=chosen_scale
            )
            ens = _rollout(fitted, d.X[int(t)], eps)
            ens_p = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(ens.shape[0], 100, -1)
            row = {"model": "D4_population_student_t", "worm_id": sid, "landmark": int(t)}
            for h in HORIZONS:
                c = _coverage(ens_p[:, h - 1], truth_p[h - 1])
                row[f"energy_h{h}"] = energy_score(ens_p[:, h - 1], truth_p[h - 1])
                row[f"crps_h{h}"] = float(crps_ensemble(ens_p[:, h - 1], truth_p[h - 1]).mean())
                for level in (50, 80, 95):
                    row[f"coverage_h{h}_{level}"] = c[f"coverage_{level}"]
                    row[f"width_h{h}_{level}"] = c[f"width_{level}"]
            row["trajectory_energy_h20"] = trajectory_energy_score(ens_p[:, :20], truth_p[:20])
            row["trajectory_energy_h100"] = trajectory_energy_score(ens_p, truth_p)
            row["neural_cov_error"] = neural_covariance_error(ens_p[:, :20], truth_p[:20])
            row["path_cov_error"] = path_covariance_error(ens_p[:, :20], truth_p[:20])
            row["autocorr_error"] = autocorrelation_error(ens_p[:, :20], truth_p[:20])
            row["mean_rmse"] = float(np.sqrt(np.mean((ens_p[:, 19].mean(0) - truth_p[19]) ** 2)))
            row["ensemble_std"] = float(ens_p[:, 19].std(0).mean())
            rows.append(row)
            for j in range(ens_p.shape[-1]):
                covd, width = interval_coverage(ens_p[:, 19], truth_p[19], 0.95)
                neuron_rows.append(
                    {
                        "model": "D4_population_student_t",
                        "worm_id": sid,
                        "landmark": int(t),
                        "neuron": j,
                        "crps_h20": crps_ensemble(ens_p[:, 19], truth_p[19])[j],
                        "coverage_50": interval_coverage(ens_p[:, 19], truth_p[19], 0.5)[0][j],
                        "coverage_80": interval_coverage(ens_p[:, 19], truth_p[19], 0.8)[0][j],
                        "coverage_95": covd[j],
                        "width_95": width[j],
                        "pit_h20": pit_values(ens_p[:, 19], truth_p[19])[j],
                    }
                )
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "d4_scores_by_landmark.csv", index=False)
    pd.DataFrame(neuron_rows).to_csv(OUT / "d4_neuron_h20.csv", index=False)
    summary = scores.agg(
        {
            "energy_h20": "mean",
            "crps_h20": "mean",
            "trajectory_energy_h20": "mean",
            "trajectory_energy_h100": "mean",
            "coverage_h20_50": "mean",
            "coverage_h20_80": "mean",
            "coverage_h20_95": "mean",
            "width_h20_95": "mean",
            "neural_cov_error": "mean",
            "path_cov_error": "mean",
            "autocorr_error": "mean",
        }
    )
    summary.to_frame().T.to_csv(OUT / "d4_summary.csv", index=False)
    print(summary.to_string())
    print("chosen scale", chosen_scale)


if __name__ == "__main__":
    main()
