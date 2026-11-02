"""Stage D-F: probabilistic adaptive VAR against strong baselines."""

from __future__ import annotations

import json

import pandas as pd

from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim
from neurobranch.probabilistic.baselines import select_baseline_config
from neurobranch.probabilistic.evaluation import (
    aggregate_pit,
    evaluate_probabilistic_session,
    load_primary,
    model_specs,
    paired_scores,
    summarize_scores,
)

OUT = PROJECT_ROOT / "reports/simulator/probabilistic_adaptive"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    _cfg, data, _recs = load_nostim()
    config, validation = select_baseline_config(data)
    validation.to_csv(OUT / "baseline_validation.csv", index=False)
    primary = load_primary()
    scores, neurons, updates, variance = [], [], [], []
    for d in data:
        sc, ne, up, va = evaluate_probabilistic_session(
            d, config, n_members=100, primary=primary, seed=17
        )
        scores.append(sc)
        neurons.append(ne)
        updates.append(up)
        variance.append(va)
        print(f"probabilistic session {d.worm_id} done: {len(sc)} rows", flush=True)
    sc = pd.concat(scores, ignore_index=True)
    ne = pd.concat(neurons, ignore_index=True)
    up = pd.concat(updates, ignore_index=True)
    va = pd.concat(variance, ignore_index=True)
    sc.to_csv(OUT / "prob_per_landmark.csv", index=False)
    ne.to_csv(OUT / "prob_per_neuron_h20.csv", index=False)
    up.to_csv(OUT / "prob_updates.csv", index=False)
    va.to_csv(OUT / "prob_variance_decomposition.csv", index=False)

    summary = summarize_scores(sc, primary=primary)
    summary.to_csv(OUT / "prob_summary.csv", index=False)
    paired = paired_scores(
        sc,
        primary=primary,
        references=("P0_persistence", "P1_local_mean", "P2_independent_ar", "P3_frozen_var"),
    )
    paired.to_csv(OUT / "prob_paired.csv", index=False)
    pit = aggregate_pit(ne)
    pit.to_csv(OUT / "prob_pit.csv", index=False)

    session_rows = []
    for (worm, model), group in sc.groupby(["worm_id", "model"]):
        session_rows.append(
            {
                "worm_id": int(worm),
                "model": model,
                "n": int(len(group)),
                "energy_h20_mean": float(group.energy_h20.mean()),
                "crps_h20_mean": float(group.crps_h20_mean.mean()),
                "coverage_50": float(group.coverage_50.mean()),
                "coverage_80": float(group.coverage_80.mean()),
                "coverage_95": float(group.coverage_95.mean()),
                "spread_error_ratio": float(group.spread_error_ratio_h20.mean()),
                "ood_h20": float(group.ood_h20.mean()),
                "temporal_covariance_error": float(group.temporal_covariance_error.mean()),
                "temporal_autocorr_error": float(group.temporal_autocorr_error.mean()),
            }
        )
    session = pd.DataFrame(session_rows)
    session.to_csv(OUT / "prob_per_session.csv", index=False)

    horizon_rows = []
    for model, group in sc.groupby("model"):
        for h in (5, 20, 50, 100):
            horizon_rows.append(
                {
                    "model": model,
                    "horizon": h,
                    "mean_energy": float(group[f"energy_h{h}"].mean()),
                    "median_energy": float(group[f"energy_h{h}"].median()),
                }
            )
    horizon = pd.DataFrame(horizon_rows)
    horizon.to_csv(OUT / "prob_horizon.csv", index=False)

    variance_summary = (
        va.groupby("model", as_index=False)
        .agg(
            within_ensemble_var=("within_ensemble_var_mean", "mean"),
            across_landmark_var=("across_landmark_mean_var_mean", "mean"),
            total_implied_var=("total_implied_var_mean", "mean"),
            observed_var=("observed_var_mean", "mean"),
            error_var=("error_var_mean", "mean"),
            spread_error_ratio=("spread_error_ratio", "mean"),
            ensemble_std_ratio=("ensemble_std_ratio", "mean"),
            total_std_ratio=("total_std_ratio", "mean"),
        )
        .sort_values("total_implied_var")
    )
    variance_summary.to_csv(OUT / "prob_variance_summary.csv", index=False)

    configs = {
        "primary_model": primary,
        "conditional_mean": "Rolling VAR w500, ridge 1.0, spectral cap 0.995",
        "n_members": 100,
        "horizons": [5, 20, 50, 100],
        "projection": "per-session training-fitted PCA, 10 components",
        "innovation_models": {
            "P0": "persistence + moving-block difference bootstrap (block 20)",
            "P1": "local mean + moving-block residual bootstrap (block 10)",
            "P2": "independent adaptive AR + diagonal Gaussian innovations",
            "P3": "frozen VAR + moving-block residual bootstrap (block 10)",
            "P4": "adaptive VAR + Ledoit-Wolf Gaussian covariance",
            "P5": "adaptive VAR + iid empirical residual bootstrap",
            "P6": "adaptive VAR + moving-block residual bootstrap (block 10)",
            "P7": "adaptive VAR + diagonal Gaussian innovations",
        },
        "baseline_config": {
            "local_window": config.local_window,
            "ewma_halflife": config.ewma_halflife,
            "ar_window": config.ar_window,
        },
        "selection_policy": "baseline settings selected on last 20% of train; innovation model selected on training residuals only",
    }
    (OUT / "prob_configurations.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")

    cards = {
        "P6_adaptive_block": {
            "type": "stochastic adaptive VAR",
            "mean": "Rolling VAR w500",
            "innovations": "moving-block empirical bootstrap, block 10",
            "status": "prespecified primary",
        },
        "P5_adaptive_boot": {
            "type": "stochastic adaptive VAR",
            "mean": "Rolling VAR w500",
            "innovations": "iid empirical bootstrap",
            "status": "secondary comparator",
        },
        "P0_persistence": {
            "type": "stochastic random walk",
            "mean": "last observation",
            "innovations": "moving-block difference bootstrap, block 20",
            "status": "simple probabilistic baseline",
        },
        "P1_local_mean": {
            "type": "stochastic local level",
            "mean": "rolling mean window 100",
            "innovations": "moving-block residual bootstrap, block 10",
            "status": "strong simple baseline",
        },
        "P2_independent_ar": {
            "type": "diagonal adaptive AR",
            "mean": "independent AR(1), window 200",
            "innovations": "diagonal Gaussian",
            "status": "no cross-neuron coupling",
        },
        "P3_frozen_var": {
            "type": "stochastic frozen VAR",
            "mean": "train-only frozen VAR",
            "innovations": "moving-block residual bootstrap, block 10",
            "status": "historical reference",
        },
        "P4_adaptive_gauss": {
            "type": "stochastic adaptive VAR",
            "mean": "Rolling VAR w500",
            "innovations": "Ledoit-Wolf Gaussian",
            "status": "innovation sensitivity",
        },
        "P7_adaptive_diag": {
            "type": "stochastic adaptive VAR",
            "mean": "Rolling VAR w500",
            "innovations": "diagonal Gaussian",
            "status": "covariance sensitivity",
        },
    }
    (OUT / "prob_model_cards.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")

    payload = {
        "stage": "D_E_F_probabilistic_adaptive",
        "git": git_state(),
        "primary_model": primary,
        "n_landmarks": int(sc[["worm_id", "landmark"]].drop_duplicates().shape[0]),
        "n_models": len(model_specs(primary)),
        "primary_score": "multivariate energy score on 10-D training PCA projection at H20, lower better",
        "secondary_scores": [
            "per-neuron CRPS",
            "50/80/95% interval coverage and width",
            "variogram score",
            "PIT histogram",
            "temporal covariance and autocorrelation error",
            "OOD frequency",
        ],
        "uncertainty": "moving-block bootstrap over overlapping landmarks; not animal-level inference",
        "evidence_tier": "exploratory repeated-session reanalysis",
    }
    (OUT / "prob_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(variance_summary.to_string(index=False))
    print(session.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
