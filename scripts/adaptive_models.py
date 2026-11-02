"""Stage C/D: frozen versus causal adaptive VAR evaluation."""

from __future__ import annotations

import json

import pandas as pd

from neurobranch.adaptive.evaluation import evaluate_session, paired_summary
from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim

OUT = PROJECT_ROOT / "reports/simulator/adaptive_dynamics"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "branch": "research/adaptive-dynamics-stability",
    "data": "Kato WT_NoStim, all five sessions",
    "standardization": "existing train-only WormData.X",
    "ridge": 1.0,
    "spectral_cap": 0.995,
    "forecast_horizons": [1, 5, 10, 20, 50, 100],
    "landmark_stride_frames": 10,
    "frozen_fit": "existing train-split-only VAR baseline, then fixed",
    "rolling_windows": [200, 500, 1000],
    "ewma_halflives": [100, 300, 1000],
    "primary_metric": "paired H20 free-running MSE improvement versus frozen VAR",
    "primary_adaptive_model": "prespecified rolling_w500; all candidates reported",
    "uncertainty": "moving-block bootstrap over prediction landmarks, block=5, 1000 draws",
}


def main() -> int:
    _cfg, data, _recs = load_nostim()
    landmark_frames, neuron_frames, update_frames, pairs = [], [], [], []
    for d in data:
        lm, neu, upd = evaluate_session(d, horizon_max=100, stride=10)
        landmark_frames.append(lm)
        neuron_frames.append(neu)
        update_frames.append(upd)
        pair = paired_summary(lm)
        pair["worm_id"] = int(d.worm_id)
        pairs.append(pair)
    landmarks = pd.concat(landmark_frames, ignore_index=True)
    neurons = pd.concat(neuron_frames, ignore_index=True)
    updates = pd.concat(update_frames, ignore_index=True)
    pairs_df = pd.concat(pairs, ignore_index=True)
    landmarks.to_csv(OUT / "model_per_landmark.csv", index=False)
    neurons.to_csv(OUT / "model_per_neuron_h20.csv", index=False)
    updates.to_csv(OUT / "model_updates.csv", index=False)
    pairs_df.to_csv(OUT / "model_paired_by_session.csv", index=False)

    h20 = landmarks[landmarks.horizon == 20]
    session = (
        h20.groupby(["worm_id", "model"], as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            median_mse=("mse", "median"),
            mean_skill_persistence=("skill_vs_persistence", "mean"),
            mean_cov_error=("covariance_relative_error", "mean"),
            mean_autocorr_error=("autocorr_error", "mean"),
            mean_std_ratio=("std_ratio", "mean"),
            mean_ood=("ood_fraction", "mean"),
        )
        .sort_values(["worm_id", "mean_mse"])
    )
    session.to_csv(OUT / "model_per_session_h20.csv", index=False)

    horizon = (
        landmarks.groupby(["model", "horizon"], as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            median_mse=("mse", "median"),
            mean_skill_persistence=("skill_vs_persistence", "mean"),
            mean_cov_error=("covariance_relative_error", "mean"),
            mean_ood=("ood_fraction", "mean"),
        )
        .sort_values(["model", "horizon"])
    )
    horizon.to_csv(OUT / "model_horizon_curves.csv", index=False)

    # Aggregate paired improvement across sessions (each landmark equal).
    pivot = h20.pivot_table(index=["worm_id", "landmark"], columns="model", values="mse")
    overall = []
    for model in [c for c in pivot.columns if c != "frozen_var"]:
        diff = (pivot["frozen_var"] - pivot[model]).dropna()
        rel = (1 - pivot[model] / pivot["frozen_var"]).dropna()
        overall.append(
            {
                "model": model,
                "n_pairs": int(len(diff)),
                "mean_mse_improvement": float(diff.mean()),
                "median_mse_improvement": float(diff.median()),
                "mean_relative_improvement": float(rel.mean()),
                "fraction_landmarks_better": float((diff > 0).mean()),
                "wins_by_session": int(
                    sum((pivot.loc[w]["frozen_var"] - pivot.loc[w][model]).dropna().mean() > 0 for w in pivot.index.get_level_values(0).unique())
                ),
            }
        )
    overall_df = pd.DataFrame(overall).sort_values("mean_mse_improvement", ascending=False)
    overall_df.to_csv(OUT / "model_paired_overall.csv", index=False)

    neuron_summary = (
        neurons.groupby(["worm_id", "model", "neuron"], as_index=False)
        .agg(
            mse_h20=("mse_h20", "mean"),
            skill_vs_persistence=("skill_vs_persistence_h20", "mean"),
            ood_rate=("ood_h20", "mean"),
        )
        .sort_values(["worm_id", "model", "skill_vs_persistence"], ascending=[True, True, False])
    )
    neuron_summary.to_csv(OUT / "model_per_neuron_summary.csv", index=False)

    update_summary = (
        updates.groupby("model", as_index=False)
        .agg(
            mean_update_seconds=("update_seconds", "mean"),
            median_update_seconds=("update_seconds", "median"),
            mean_spectral_radius=("spectral_radius", "mean"),
            max_spectral_radius=("spectral_radius", "max"),
            clip_fraction=("clip_scale", lambda s: float((s < 1).mean())),
            mean_n_transitions=("n_transitions", "mean"),
        )
        .sort_values("mean_update_seconds", ascending=False)
    )
    update_summary.to_csv(OUT / "model_update_summary.csv", index=False)

    (OUT / "model_configurations.json").write_text(
        json.dumps(CONFIG, indent=2), encoding="utf-8"
    )
    cards = {
        "C1_frozen_var": {
            "class": "FrozenVAR",
            "fit": "weighted ridge lag-one VAR on all pre-test transitions; coefficients frozen",
            "parameters": "full 103x103 A and 103 intercepts per session",
            "stability": "spectral radius capped at 0.995 by uniform scaling",
        },
        "C2_rolling_var": {
            "class": "RollingVAR",
            "history_windows_frames": [200, 500, 1000],
            "fit": "equal-weight ridge on transitions with target <= landmark",
            "stability": "same spectral cap at every refit",
            "primary": "rolling_w500",
        },
        "C3_ewma_var": {
            "class": "EWMAVAR",
            "halflives_frames": [100, 300, 1000],
            "fit": "exponentially weighted ridge on all transitions <= landmark; half-life fixed",
            "stability": "same spectral cap at every refit",
        },
        "C4_local_latent": {
            "status": "not fitted",
            "reason": "Stage A found no stable train-only latent orientation; adding a local latent encoder would mix incompatible coordinate systems and is not justified before C1-C3 are resolved.",
        },
    }
    (OUT / "model_cards.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")
    payload = {
        "stage": "C_D_adaptive_models",
        "config": CONFIG,
        "paired_overall": overall_df.to_dict(orient="records"),
        "paired_by_session": pairs_df.to_dict(orient="records"),
        "updates": update_summary.to_dict(orient="records"),
        "git": git_state(),
    }
    (OUT / "model_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(overall_df.to_string(index=False))
    print(horizon[horizon.horizon == 20].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
