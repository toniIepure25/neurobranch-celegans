"""Stage A: reproduce the deliberately overdispersed historical population model.

This script is diagnostic only.  It admits multi-step forecast errors to the
innovation pool exactly as the historical model did; the resulting artifacts
must never be used as a candidate generative model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from neurobranch.adaptive.evaluation import evaluation_landmarks
from neurobranch.adaptive.var import RollingVAR
from neurobranch.diagnostics.common import load_nostim
from neurobranch.probabilistic.metrics import coverage_summary, crps_ensemble, energy_score
from neurobranch.probabilistic.residual import adaptive_prequential_residuals
from neurobranch.probabilistic.stochastic import build_innovation, ensemble_rollout
from neurobranch.uncertainty_calibration.trajectory import trajectory_energy_score

OUT = Path("reports/simulator/joint_dynamics")


def main() -> None:
    _, data, _ = load_nostim()
    rows = []
    for d in data:
        sid = int(d.worm_id)
        pca = PCA(n_components=10, random_state=17).fit(d.X[d.masks["train"]])
        series = adaptive_prequential_residuals(d, window=500, block=50)
        for t in evaluation_landmarks(d, horizon_max=100, stride=10):
            t = int(t)
            fitted = RollingVAR(window=500).fit_at(d.X, t)
            if fitted is None:
                continue
            pool = series.values[series.targets <= t][-1000:]
            innovation = build_innovation(pool, "block", block=10)
            ens = ensemble_rollout(
                "var", fitted, d.X[t], 100, 100, innovation, np.random.default_rng(7000 + sid * 10000 + t)
            )
            ens_p = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(ens.shape[0], 100, -1)
            truth = pca.transform(d.X[t + 1 : t + 101])
            truth_raw = d.X[t + 1 : t + 101]
            c = coverage_summary(ens_p[:, 19], truth[19])
            cf = coverage_summary(ens[:, 19], truth_raw[19])
            rows.append(
                {
                    "model": "P6_historical_block",
                    "worm_id": sid,
                    "landmark": t,
                    "energy_h20": energy_score(ens_p[:, 19], truth[19]),
                    "crps_h20": float(crps_ensemble(ens_p[:, 19], truth[19]).mean()),
                    "crps_h20_full": float(crps_ensemble(ens[:, 19], truth_raw[19]).mean()),
                    "coverage_50": c["coverage_50"],
                    "coverage_80": c["coverage_80"],
                    "coverage_95": c["coverage_95"],
                    "coverage_full_50": cf["coverage_50"],
                    "coverage_full_80": cf["coverage_80"],
                    "coverage_full_95": cf["coverage_95"],
                    "width_95": c["width_95"],
                    "trajectory_energy_h20": trajectory_energy_score(ens_p[:, :20], truth[:20]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "historical_block_scores.csv", index=False)
    print(df[["energy_h20", "crps_h20", "coverage_50", "coverage_80", "coverage_95", "trajectory_energy_h20"]].mean().to_string())


if __name__ == "__main__":
    main()
