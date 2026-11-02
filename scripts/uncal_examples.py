"""Executed examples for the uncertainty dashboard and figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.adaptive.evaluation import evaluation_landmarks
from neurobranch.adaptive.var import RollingVAR
from neurobranch.diagnostics.common import load_nostim
from neurobranch.probabilistic.baselines import IndependentARModel
from neurobranch.probabilistic.residual import adaptive_prequential_residuals, independent_ar_residuals
from neurobranch.probabilistic.stochastic import build_innovation, ensemble_rollout

OUT = Path("reports/simulator/uncertainty_calibration")


def main() -> None:
    cfg, data, _ = load_nostim()
    d = data[0]
    t = int(evaluation_landmarks(d, horizon_max=20, stride=10)[6])
    ar = IndependentARModel(window=200).fit_at(d.X, t)
    ind = independent_ar_residuals(d, window=200, block=10)
    hist = adaptive_prequential_residuals(d, window=500, block=50)
    lead = ind.values[(ind.targets <= t) & (ind.lead == 1)][-300:]
    models = {}
    models["P2_independent_ar_lead1"] = ensemble_rollout(
        "independent_ar", ar, d.X[t], 20, 200,
        build_innovation(lead, "gauss_diag"), np.random.default_rng(123),
    )
    p0_pool = d.X[np.flatnonzero(d.masks["train"])[0] + 1 : t + 1] - d.X[np.flatnonzero(d.masks["train"])[0] : t]
    models["P0_persistence"] = ensemble_rollout(
        "persistence", None, d.X[t], 20, 200,
        build_innovation(p0_pool[-1000:], "block", block=20), np.random.default_rng(124),
    )
    adaptive = hist.values[hist.targets <= t][-1000:]
    models["P6_adaptive_block"] = ensemble_rollout(
        "var", RollingVAR(window=500).fit_at(d.X, t),
        d.X[t], 20, 200, build_innovation(adaptive, "block", block=10), np.random.default_rng(125),
    )
    lead_adaptive = hist.values[(hist.targets <= t) & (hist.lead == 1)][-200:]
    models["P6_adaptive_lead1"] = ensemble_rollout(
        "var", RollingVAR(window=500).fit_at(d.X, t),
        d.X[t], 20, 200, build_innovation(lead_adaptive, "block", block=10), np.random.default_rng(126),
    )
    truth = d.X[t : t + 20]
    rows, bands, observed = [], [], []
    selected = [0, 1, 2, 5]
    for model, ens in models.items():
        pop = ens.mean(axis=2)
        for member in range(ens.shape[0]):
            for step in range(20):
                rows.append({"worm_id": int(d.worm_id), "landmark": t, "model": model, "member": member, "step": step, "series": "population_mean", "value": float(pop[member, step])})
                for j in selected:
                    rows.append({"worm_id": int(d.worm_id), "landmark": t, "model": model, "member": member, "step": step, "series": d.predictor_names[j], "value": float(ens[member, step, j])})
        for step in range(20):
            bands.append({"model": model, "step": step, "series": "population_mean", "q05": float(np.quantile(pop[:, step], 0.05)), "q50": float(np.quantile(pop[:, step], 0.5)), "q95": float(np.quantile(pop[:, step], 0.95))})
            for j in selected:
                bands.append({"model": model, "step": step, "series": d.predictor_names[j], "q05": float(np.quantile(ens[:, step, j], 0.05)), "q50": float(np.quantile(ens[:, step, j], 0.5)), "q95": float(np.quantile(ens[:, step, j], 0.95))})
    for step in range(20):
        observed.append({"step": step, "series": "population_mean", "value": float(truth[step].mean())})
        for j in selected:
            observed.append({"step": step, "series": d.predictor_names[j], "value": float(truth[step, j])})
    pd.DataFrame(rows).to_csv(OUT / "example_trajectories.csv", index=False)
    pd.DataFrame(bands).to_csv(OUT / "example_bands.csv", index=False)
    pd.DataFrame(observed).to_csv(OUT / "example_observed.csv", index=False)
    (OUT / "example_metadata.json").write_text(
        json.dumps(
            {
                "worm_id": int(d.worm_id),
                "landmark": t,
                "horizon": 20,
                "members": 200,
                "selected_neurons": [d.predictor_names[j] for j in selected],
                "models": list(models),
                "note": "Executed ensembles; not decorative trajectories.",
            },
            indent=2,
        )
    )
    print("example", t, len(rows), len(bands))


if __name__ == "__main__":
    main()
