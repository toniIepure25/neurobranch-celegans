"""Metrics and block-aware paired uncertainty for the interaction audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.joint_dynamics.evaluation import (
    autocorrelation_error,
    neural_covariance_error,
    path_covariance_error,
)
from neurobranch.probabilistic.metrics import coverage_summary, energy_score
from neurobranch.uncertainty_calibration.trajectory import trajectory_energy_score


def ensemble_metrics(ensemble: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """H20 metrics shared by every mean-dynamics comparison."""
    ens = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    h = min(20, ens.shape[1], truth.shape[0])
    cov = coverage_summary(ens[:, h - 1], truth[h - 1])
    return {
        "energy_h20": float(energy_score(ens[:, h - 1], truth[h - 1])),
        "trajectory_energy_h20": float(trajectory_energy_score(ens[:, :h], truth[:h])),
        "mse_mean_h20": float(np.mean((ens[:, h - 1].mean(axis=0) - truth[h - 1]) ** 2)),
        "coverage_50": float(cov["coverage_50"]),
        "coverage_80": float(cov["coverage_80"]),
        "coverage_95": float(cov["coverage_95"]),
        "neural_cov_error": float(neural_covariance_error(ens[:, :h], truth[:h])),
        "path_cov_error": float(path_covariance_error(ens[:, :h], truth[:h])),
        "autocorr_error": float(autocorrelation_error(ens[:, :h], truth[:h])),
    }


def session_block_ci(
    frame: pd.DataFrame,
    value: str,
    *,
    session: str = "worm_id",
    block: int = 5,
    n_boot: int = 1000,
    seed: int = 17,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Moving-block bootstrap within each session, then equal-session pooling.

    Blocks never cross a recording boundary.  The resulting interval is a
    repeated-session interval, not an animal-level interval.
    """
    work = frame[[session, "landmark", value]].dropna().sort_values([session, "landmark"])
    if work.empty:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(int(seed))
    groups = [g[value].to_numpy(dtype=np.float64) for _, g in work.groupby(session, sort=True)]
    groups = [g for g in groups if len(g)]
    if not groups:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    means = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        sampled = []
        for values in groups:
            n = len(values)
            bl = max(1, min(int(block), n))
            n_blocks = int(np.ceil(n / bl))
            starts = rng.integers(0, n - bl + 1, size=n_blocks)
            sampled.append(np.concatenate([values[s : s + bl] for s in starts])[:n])
        means[b] = float(np.concatenate(sampled).mean())
    all_values = np.concatenate(groups)
    return {
        "mean": float(all_values.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(len(all_values)),
        "block": int(block),
    }


def paired_row(
    frame: pd.DataFrame,
    *,
    reference: str,
    model: str,
    metric: str,
    block: int = 5,
    n_boot: int = 1000,
    seed: int = 17,
) -> dict[str, object]:
    """Paired improvement ``reference - model`` for one metric."""
    pivot = frame.pivot_table(
        index=["worm_id", "landmark"],
        columns="model",
        values=metric,
    )
    if reference not in pivot or model not in pivot:
        return {}
    diff = (pivot[reference] - pivot[model]).dropna().rename("improvement").reset_index()
    ci = session_block_ci(diff, "improvement", block=block, n_boot=n_boot, seed=seed)
    return {
        "model": model,
        "reference": reference,
        "metric": metric,
        "n_pairs": ci["n"],
        "mean_improvement": ci["mean"],
        "ci_low": ci["lo"],
        "ci_high": ci["hi"],
        "fraction_better": float((diff["improvement"] > 0).mean()) if len(diff) else float("nan"),
        "block": int(block),
    }


def paired_table(
    frame: pd.DataFrame,
    *,
    reference: str,
    models: list[str],
    metric: str,
    blocks: tuple[int, ...] = (3, 5, 10),
    n_boot: int = 1000,
) -> pd.DataFrame:
    rows = []
    for block in blocks:
        for model in models:
            row = paired_row(
                frame,
                reference=reference,
                model=model,
                metric=metric,
                block=block,
                n_boot=n_boot,
                seed=1700 + block,
            )
            if row:
                rows.append(row)
    return pd.DataFrame(rows)
