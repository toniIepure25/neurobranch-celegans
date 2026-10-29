"""Tests for the descriptive fold-level leaderboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.benchmark.leaderboard import build_leaderboard, leaderboard_summary


def _metrics() -> pd.DataFrame:
    rows = []
    for fold in range(3):
        for seed in range(2):
            rows.append(
                {
                    "model": "twin",
                    "k": 8,
                    "strategy": "random",
                    "fold": fold,
                    "model_seed": seed,
                    "acquisition_seed": 0,
                    "pair_spearman": 0.10 + 0.01 * fold,
                    "profile_spearman": 0.09,
                    "coverage": 0.9,
                    "interval_score": 30.0,
                }
            )
        rows.append(
            {
                "model": "cro_static",
                "k": 0,
                "strategy": "none",
                "fold": fold,
                "model_seed": 0,
                "acquisition_seed": 0,
                "pair_spearman": 0.02,
                "profile_spearman": 0.02,
                "coverage": 0.9,
                "interval_score": 30.0,
            }
        )
    return pd.DataFrame(rows)


def test_leaderboard_pairs_per_fold_not_per_seed() -> None:
    frame = build_leaderboard(_metrics(), ceiling=0.8)
    twin = frame[frame["model"] == "twin"].iloc[0]
    assert twin["n_folds"] == 3
    assert twin["n_paired_folds"] == 3
    assert np.isclose(twin["delta_vs_baseline"], twin["pair_spearman_mean"] - 0.02)
    assert twin["unit"] == "fold"
    assert twin["status"] == "descriptive"


def test_leaderboard_summary_keeps_fold_unit() -> None:
    summary = leaderboard_summary(build_leaderboard(_metrics(), ceiling=0.8))
    assert summary["unit"] == "fold"
    assert summary["n_configurations"] == 2
    assert summary["best_model"] == "twin"
