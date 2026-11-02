"""Stage G: one observed history, stochastic rollouts and predictive bands."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from neurobranch.adaptive.evaluation import evaluation_landmarks
from neurobranch.adaptive.var import RollingVAR
from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim
from neurobranch.probabilistic.baselines import LocalMeanModel
from neurobranch.probabilistic.evaluation import load_primary
from neurobranch.probabilistic.residual import (
    adaptive_prequential_residuals,
    local_mean_residuals,
)
from neurobranch.probabilistic.stochastic import build_innovation, ensemble_rollout

OUT = PROJECT_ROOT / "reports/simulator/probabilistic_adaptive"
OUT.mkdir(parents=True, exist_ok=True)
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def _population(ensemble: np.ndarray) -> np.ndarray:
    return ensemble.mean(axis=-1)


def main() -> int:
    _cfg, data, _recs = load_nostim()
    d = data[0]
    landmarks = evaluation_landmarks(d, horizon_max=100, stride=10)
    t = int(landmarks[len(landmarks) // 3])
    horizon = 20
    x0 = d.X[t]
    truth = d.X[t + 1 : t + horizon + 1]
    pca = PCA(n_components=10, random_state=17).fit(d.X[d.masks["train"]])
    rng = np.random.default_rng(1234)
    adaptive = RollingVAR(window=500).fit_at(d.X, t)
    local = LocalMeanModel(window=100).fit_at(d.X, t)
    adaptive_res = adaptive_prequential_residuals(d, window=500)
    local_res = local_mean_residuals(d, window=100)
    pool_adapt = adaptive_res.subset(t)
    pool_local = local_res.subset(t)
    primary = load_primary()
    block = 10 if primary == "P6_adaptive_block" else 5
    ens_adapt = ensemble_rollout(
        "var",
        adaptive,
        x0,
        horizon,
        100,
        build_innovation(pool_adapt, "block", block=block),
        rng,
    )
    ens_local = ensemble_rollout(
        "local_mean",
        local,
        x0,
        horizon,
        100,
        build_innovation(pool_local, "block", block=10),
        rng,
    )
    det = adaptive.forecast(x0, horizon)
    det_local = local.forecast(x0, horizon)

    trajectory_rows = []
    for sample in range(20):
        pop = _population(ens_adapt[sample])
        latent = pca.transform(ens_adapt[sample])
        for step in range(horizon):
            trajectory_rows.append(
                {
                    "sample": sample,
                    "step": step + 1,
                    "population_mean": float(pop[step]),
                    "pc1": float(latent[step, 0]),
                    "pc2": float(latent[step, 1]),
                }
            )
    pd.DataFrame(trajectory_rows).to_csv(OUT / "prob_example_trajectories.csv", index=False)

    band_rows = []
    for name, ens, det_pred in (
        ("P6_adaptive_block", ens_adapt, det),
        ("P1_local_mean", ens_local, det_local),
    ):
        pop = _population(ens)
        q05, q50, q95 = np.quantile(pop, [0.05, 0.5, 0.95], axis=0)
        for step in range(horizon):
            band_rows.append(
                {
                    "model": name,
                    "step": step + 1,
                    "deterministic_population_mean": float(_population(det_pred[None, :, :])[0, step]),
                    "ensemble_population_mean": float(pop[:, step].mean()),
                    "q05": float(q05[step]),
                    "q50": float(q50[step]),
                    "q95": float(q95[step]),
                    "truth_population_mean": float(_population(truth[None, :, :])[0, step]),
                }
            )
    bands = pd.DataFrame(band_rows)
    bands.to_csv(OUT / "prob_example_bands.csv", index=False)

    diag = []
    for name, ens, det_pred in (
        ("P6_adaptive_block", ens_adapt, det),
        ("P1_local_mean", ens_local, det_local),
    ):
        pop = _population(ens)
        final_spread = float(pop[:, -1].std(ddof=1))
        final_error = float(abs(pop[:, -1].mean() - truth[-1].mean()))
        diag.append(
            {
                "model": name,
                "worm_id": int(d.worm_id),
                "landmark": t,
                "ensemble_population_std_h20": final_spread,
                "abs_mean_error_h20": final_error,
                "spread_over_error": final_spread / max(final_error, 1e-12),
                "deterministic_mse_h20": float(np.mean((det_pred[-1] - truth[-1]) ** 2)),
                "ensemble_mean_mse_h20": float(np.mean((ens[:, -1, :].mean(axis=0) - truth[-1]) ** 2)),
            }
        )
    diagnostics = pd.DataFrame(diag)
    diagnostics.to_csv(OUT / "prob_example_diagnostics.csv", index=False)
    payload = {
        "stage": "G_trajectory_example",
        "worm_id": int(d.worm_id),
        "landmark": t,
        "horizon": horizon,
        "n_stored_trajectories": 20,
        "n_ensemble": 100,
        "primary_model": primary,
        "projection": "per-session training PCA, first two components for visualization only",
        "warning": WARNING,
        "note": "Divergent simulated paths are not biological bifurcations.",
        "git": git_state(),
    }
    (OUT / "prob_example_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(diagnostics.to_string(index=False))
    print(f"example landmark {t}, stored {len(trajectory_rows)} trajectory rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
