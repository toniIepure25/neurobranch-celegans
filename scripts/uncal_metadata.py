"""Model cards, configurations and reproducibility metadata."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("reports/simulator/uncertainty_calibration")

CONFIGURATIONS = {
    "branch": "research/neural-uncertainty-calibration",
    "starting_head": "2c94fe7c1120a7536367ac962f8586797969f5c9",
    "data": "Kato WT_NoStim, five repeated sessions, existing train-standardized X",
    "split": {
        "train_frames": "0:2195",
        "validation_frames": "2207:2677",
        "test_frames": "2689:3137",
        "test_landmarks": "split_starts(test, history=2, horizon=100)[::10]",
        "n_landmarks": 175,
        "horizons": [1, 5, 10, 20, 50, 100],
        "primary_horizon": 20,
    },
    "ensembles": {"n_members": 100, "seed_formula": "7000 + worm_id*10000 + landmark", "pca_components": 10},
    "baselines": {
        "P0_persistence": {"innovation": "moving-block bootstrap", "block": 20, "history": 1000},
        "P1_local_mean": {"window": 100, "innovation": "moving-block bootstrap", "block": 10, "history": 1000},
        "P6_adaptive_block": {"mean": "rolling VAR w500", "residual_block": 50, "innovation": "moving-block bootstrap", "innovation_block": 10, "history": 1000},
        "P6_adaptive_lead1": {"mean": "rolling VAR w500", "residual_block": 10, "innovation": "moving-block bootstrap", "innovation_block": 10, "lead": 1, "history": 1000},
    },
    "primary": {
        "name": "P2_independent_ar_lead1",
        "mean": "independent per-neuron AR(1)",
        "window": 200,
        "ridge": 1.0,
        "spectral_cap": 0.995,
        "residual_block": 10,
        "innovation_lead": 1,
        "innovation": "diagonal Gaussian",
        "scale_history": 1000,
    },
    "secondary": {
        "P2_fixed_train_scale": {"scale": "training lead-1 residuals only"},
        "P3_state_ar": {"variance_features": ["abs_level", "abs_change", "recent_std", "recent_abs_mean"], "ridge": 1.0, "clip": [0.01, 3.0]},
    },
    "scores": {
        "primary": "mean per-neuron CRPS at H20",
        "secondary": ["energy", "CRPS", "variogram", "PIT", "coverage", "width", "variance ratio", "covariance error", "autocorrelation error", "OOD"],
        "trajectory_representation": "flattened H20 x 10-D training PCA, normalized by sqrt(20)",
    },
    "uncertainty": {"block_bootstrap": 5, "n_boot": 1000},
}

CARDS = [
    {
        "model": "P0_persistence",
        "role": "frozen baseline",
        "mean": "last observed state",
        "innovation": "moving-block bootstrap of historical differences",
        "parameters": {"block": 20},
        "known_limitations": "random-walk uncertainty; no mean reversion",
    },
    {
        "model": "P1_local_mean",
        "role": "frozen baseline",
        "mean": "rolling local mean, window 100",
        "innovation": "moving-block bootstrap of local-mean residuals",
        "parameters": {"block": 10},
        "known_limitations": "constant forecast; no temporal response",
    },
    {
        "model": "P2_independent_ar_lead1",
        "role": "prespecified primary",
        "mean": "independent per-neuron AR(1), window 200, ridge 1.0, cap 0.995",
        "innovation": "regularized diagonal Gaussian from causal lead-1 block-prequential residuals",
        "parameters": {"residual_block": 10, "history": 1000, "members": 100},
        "known_limitations": "no cross-neuron covariance; under-covers at 95%; not a generative trajectory model",
    },
    {
        "model": "P2_fixed_train_scale",
        "role": "sensitivity",
        "mean": "same as primary",
        "innovation": "diagonal scale fixed from training residuals only",
        "parameters": {},
        "known_limitations": "no online adaptation; otherwise identical",
    },
    {
        "model": "P3_state_ar",
        "role": "prespecified secondary",
        "mean": "same as primary",
        "innovation": "state-conditioned diagonal Gaussian",
        "parameters": {"features": 4, "ridge": 1.0},
        "known_limitations": "state-only scale; overconfident in low-variance states",
    },
    {
        "model": "P6_adaptive_block",
        "role": "historical frozen comparator",
        "mean": "rolling population VAR, window 500",
        "innovation": "moving-block bootstrap of block-50 residuals, leads 1-50 pooled",
        "parameters": {"block": 10},
        "known_limitations": "innovation pool is not one-step innovations; severely overdispersed",
    },
    {
        "model": "P6_adaptive_lead1",
        "role": "diagnostic correction",
        "mean": "rolling population VAR, window 500",
        "innovation": "moving-block bootstrap of lead-1 residuals",
        "parameters": {"block": 10},
        "known_limitations": "diagnostic only; 50% intervals over-cover; joint covariance still poor",
    },
    {
        "model": "D4_scaled_diag",
        "role": "diagnostic calibration baseline",
        "mean": "same as primary",
        "innovation": "diagonal Gaussian with validation-fitted global variance factor",
        "parameters": {"factor_source": "validation split only"},
        "known_limitations": "factor unstable across sessions; over-inflates test intervals; not mechanistic",
    },
]


def main() -> None:
    (OUT / "configurations.json").write_text(json.dumps(CONFIGURATIONS, indent=2))
    (OUT / "model_cards.json").write_text(json.dumps(CARDS, indent=2))
    (OUT / "gate_thresholds.json").write_text(
        json.dumps(
            {
                "D": "H20 CRPS improvement over P0 and P1 with paired 95% bootstrap lower bound > 0",
                "E": {
                    "coverage_50": 0.07,
                    "coverage_80": 0.07,
                    "coverage_95": 0.05,
                    "pit_error_max": 0.05,
                    "width_ratio_range": [0.3, 2.0],
                    "standardized_error_variance_range": [0.7, 1.4],
                    "max_horizon_deviation": 0.12,
                },
                "F": {
                    "trajectory_improvement_ci_low": 0.0,
                    "total_std_ratio_range": [0.75, 1.35],
                    "autocorrelation_error_max": 0.25,
                    "covariance_relative_error_max": 0.5,
                    "ood_max": 0.2,
                },
                "frozen_before_final_evaluation": True,
            },
            indent=2,
        )
    )
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "study": "Neural uncertainty decomposition and calibrated forecasts",
                "status": "exploratory repeated-session reanalysis",
                "historical_reports_frozen": True,
                "raw_recordings_committed": False,
                "large_weights_committed": False,
                "reproduction_commands": [
                    "python scripts/uncal_diagnostics.py",
                    "python scripts/uncal_evaluation.py",
                    "python scripts/uncal_d4.py",
                    "python scripts/uncal_lowrank.py",
                    "python scripts/uncal_synthetic.py",
                    "python scripts/uncal_reproduction.py",
                    "python scripts/uncal_summarize.py",
                    "python scripts/uncal_examples.py",
                    "python scripts/uncal_figures.py",
                    "python scripts/uncal_manifest.py",
                    "python scripts/validate_uncal_reports.py",
                ],
                "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
                "biological_validation": "NOT_EVALUABLE",
                "repeated_sessions_not_animals": True,
                "landmark_overlap": "H20 landmarks overlap by 10 frames; H50/H100 overlap more",
            },
            indent=2,
        )
    )
    print("metadata written")


if __name__ == "__main__":
    main()
