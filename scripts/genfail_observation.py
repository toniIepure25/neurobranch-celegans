"""GENFAIL Stage C: synthetic controls and real observation-model comparison."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import (
    block_bootstrap_skill,
    load_nostim,
    skill_vs_persistence,
)
from neurobranch.diagnostics.observation import CausalCalciumLDS, SecondOrderLatentLDS
from neurobranch.diagnostics.synthetic import run as run_synthetic
from neurobranch.dynamics.context_models import StableLatentLDS
from neurobranch.preprocessing import split_starts

OUT = PROJECT_ROOT / "reports/simulator/failure_diagnostics/observation"
HORIZONS = (1, 20, 100)


def _real(data, *, latent_dim: int = 10) -> tuple[list[dict], list[dict]]:
    train1 = split_starts(data, "train", history=1, horizon=1)
    train2 = split_starts(data, "train", history=2, horizon=1)
    labels = data.y_macro
    ar1 = StableLatentLDS.fit(
        data.X,
        labels,
        train1,
        train_mask=data.masks["train"],
        latent_dim=latent_dim,
        ridge=1e-2,
    )
    ar2 = SecondOrderLatentLDS.fit(
        data.X, data.masks["train"], train2, latent_dim=latent_dim, ridge=1e-2
    )
    calcium = CausalCalciumLDS.fit(
        data.X, data.masks["train"], train2, latent_dim=latent_dim, ridge=1e-2
    )
    rows = []
    alpha_rows = [
        {
            "worm_id": int(data.worm_id),
            "alpha": float(item["alpha"]),
            "train_one_step_mse": float(item["train_one_step_mse"]),
        }
        for item in calcium.alpha_table
    ]
    for horizon in HORIZONS:
        starts = split_starts(data, "test", history=2, horizon=horizon)
        hist = np.stack([data.X[t - 1 : t + 1] for t in starts])
        truth = np.stack([data.X[t + 1 : t + horizon + 1] for t in starts])
        persistence = np.repeat(hist[:, -1:, :], horizon, axis=1)
        base = float(np.mean((persistence - truth) ** 2))
        preds = {}
        if horizon == 1:
            preds["ar1"] = ar1.predict_next(hist)[:, None, :]
            preds["ar2"] = ar2.predict_next(hist)[:, None, :]
            preds["calcium"] = calcium.predict_next(hist)[:, None, :]
        else:
            preds["ar1"] = np.stack(
                [ar1.rollout(np.zeros(data.n_neurons), horizon, initial_history=h)[1:] for h in hist]
            )
            preds["ar2"] = np.stack(
                [ar2.rollout(np.zeros(data.n_neurons), horizon, initial_history=h)[1:] for h in hist]
            )
            preds["calcium"] = np.stack(
                [calcium.rollout(np.zeros(data.n_neurons), horizon, initial_history=h)[1:] for h in hist]
            )
        for name, pred in preds.items():
            mse = float(np.mean((pred - truth) ** 2))
            per_start = ((pred - truth) ** 2).mean(axis=(1, 2))
            per_base = ((persistence - truth) ** 2).mean(axis=(1, 2))
            ci = block_bootstrap_skill(per_start, per_base, n_boot=400)
            rows.append(
                {
                    "model": name,
                    "worm_id": int(data.worm_id),
                    "horizon": int(horizon),
                    "n_starts": int(len(starts)),
                    "mse": mse,
                    "persistence_mse": base,
                    "skill_vs_persistence": skill_vs_persistence(mse, base),
                    "skill_ci_low": ci["lo"],
                    "skill_ci_high": ci["hi"],
                    "latent_dim": int(latent_dim),
                    "selected_alpha": float(calcium.alpha) if name == "calcium" else float("nan"),
                }
            )
    return rows, alpha_rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    synth, latent, perturb = run_synthetic()
    synth.to_csv(OUT / "synthetic_forecast.csv", index=False)
    latent.to_csv(OUT / "synthetic_latent_recovery.csv", index=False)
    perturb.to_csv(OUT / "synthetic_perturbation.csv", index=False)

    cfg, datasets, _ = load_nostim()
    rows, alpha_rows = [], []
    for data in datasets:
        print(f"observation model worm {data.worm_id}", flush=True)
        r, a = _real(data)
        rows.extend(r)
        alpha_rows.extend(a)
    real = pd.DataFrame(rows)
    alpha = pd.DataFrame(alpha_rows)
    real.to_csv(OUT / "real_observation_models.csv", index=False)
    alpha.to_csv(OUT / "real_alpha_selection.csv", index=False)
    agg = (
        real.groupby(["model", "horizon"], as_index=False)
        .agg(
            mean_skill=("skill_vs_persistence", "mean"),
            median_skill=("skill_vs_persistence", "median"),
            min_skill=("skill_vs_persistence", "min"),
            max_skill=("skill_vs_persistence", "max"),
            mean_mse=("mse", "mean"),
        )
        .sort_values(["horizon", "mean_skill"], ascending=[True, False])
    )
    agg.to_csv(OUT / "real_observation_aggregate.csv", index=False)
    metadata = {
        "stage": "C_observation_model",
        "evidence_tier": "exploratory",
        "data_sha256": datasets[0].recording.source_sha256,
        "latent_dim": 10,
        "models": ["ar1_latent_lds", "second_order_latent_lds", "causal_calcium_lds"],
        "selection": "alpha selected on train one-step MSE; no test selection",
        "synthetic_conditions": sorted(synth.condition.unique().tolist()),
        "synthetic_note": "synthetic success is not biological validation",
        "git": git_state(),
    }
    (OUT / "observation_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(agg.to_string(index=False))
    print(synth.groupby(["condition", "model"]).skill_h20.mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
