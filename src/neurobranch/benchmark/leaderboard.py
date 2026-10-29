"""Descriptive leaderboard for the Phase 8 fold-level metrics.

The unit is a **fold**, not an animal.  Folds share animals and responders, so
the fold-level spread is not an animal-generalisation interval.  The table is a
navigation aid, not a ranking claim; the claims ledger carries the inferential
status.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

METRICS = ("pair_spearman", "profile_spearman", "coverage", "interval_score")
BASELINE = "cro_static"


def build_leaderboard(
    metrics: pd.DataFrame,
    *,
    ceiling: float,
    baseline: str = BASELINE,
) -> pd.DataFrame:
    """Aggregate model configurations and attach a paired baseline delta.

    Args:
        metrics: Phase 8 ``metrics.csv`` with one row per fold and configuration.
        ceiling: pooled reliability ceiling used only as a descriptive ratio.
        baseline: model name used for the paired fold delta.
    """
    required = {"model", "k", "strategy", "fold", *METRICS}
    missing = required - set(metrics.columns)
    if missing:
        raise KeyError(f"metrics missing columns: {sorted(missing)}")
    if ceiling <= 0:
        raise ValueError("ceiling must be positive")

    grouped = metrics.groupby(["model", "k", "strategy"], dropna=False)
    rows: list[dict[str, Any]] = []
    baseline_rows = metrics[metrics["model"] == baseline]
    baseline_by_fold = (
        baseline_rows.groupby("fold")["pair_spearman"].mean() if not baseline_rows.empty else None
    )
    for (model, k, strategy), group in grouped:
        pair = float(group["pair_spearman"].mean()) if group["pair_spearman"].notna().any() else float("nan")
        delta = float("nan")
        delta_low = float("nan")
        delta_high = float("nan")
        n_paired = 0
        if baseline_by_fold is not None and model != baseline:
            fold_means = group.groupby("fold")["pair_spearman"].mean()
            paired = fold_means - baseline_by_fold
            paired = paired.dropna()
            n_paired = int(len(paired))
            if n_paired:
                delta = float(paired.mean())
                # Fold-level spread only: folds are not independent replicates.
                delta_low = float(paired.min())
                delta_high = float(paired.max())
        rows.append(
            {
                "model": model,
                "k": int(k) if pd.notna(k) else 0,
                "strategy": strategy,
                "n_folds": int(group["fold"].nunique()),
                "pair_spearman_mean": pair,
                "profile_spearman_mean": float(group["profile_spearman"].mean()),
                "coverage_mean": float(group["coverage"].mean()),
                "interval_score_mean": float(group["interval_score"].mean()),
                "ceiling_ratio": pair / ceiling if np.isfinite(pair) else float("nan"),
                "delta_vs_baseline": delta,
                "delta_fold_min": delta_low,
                "delta_fold_max": delta_high,
                "n_paired_folds": n_paired,
                "unit": "fold",
                "status": "descriptive",
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["pair_spearman_mean", "model", "k", "strategy"], ascending=[False, True, True, True]
    ).reset_index(drop=True)


def leaderboard_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Small JSON-friendly summary that never upgrades the unit to animals."""
    if frame.empty:
        return {"n_configurations": 0, "unit": "fold", "best_model": None}
    best = frame.iloc[0]
    twin = frame[frame["model"] == "twin"]
    return {
        "n_configurations": int(len(frame)),
        "unit": "fold",
        "best_model": str(best["model"]),
        "best_k": int(best["k"]),
        "best_strategy": str(best["strategy"]),
        "best_pair_spearman": float(best["pair_spearman_mean"]),
        "best_ceiling_ratio": float(best["ceiling_ratio"]),
        "twin_max_pair_spearman": float(twin["pair_spearman_mean"].max()) if len(twin) else None,
        "caveat": "Folds share animals; deltas are descriptive, not animal-level intervals.",
    }


__all__ = ["BASELINE", "METRICS", "build_leaderboard", "leaderboard_summary"]
