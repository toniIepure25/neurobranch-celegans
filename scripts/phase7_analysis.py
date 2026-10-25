"""Aggregate Phase 7 budget curves, paired deltas and preregistered decisions."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, resolve_path


def _bootstrap_ci(values: np.ndarray, *, n_boot: int = 2000, seed: int = 17) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=values.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_delta(
    frame: pd.DataFrame,
    *,
    strategy: str,
    budget: float,
    reference: str = "random",
) -> dict[str, float]:
    subset = frame[np.isclose(frame["budget"], budget)]
    keys = ["fold", "acquisition_seed"]
    a = subset[subset["strategy"] == strategy][keys + ["pair_spearman"]].rename(
        columns={"pair_spearman": "strategy"}
    )
    b = subset[subset["strategy"] == reference][keys + ["pair_spearman"]].rename(
        columns={"pair_spearman": "reference"}
    )
    merged = a.merge(b, on=keys, how="inner").dropna()
    if merged.empty:
        return {
            "strategy": strategy,
            "reference": reference,
            "budget": budget,
            "n_blocks": 0,
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "wins": 0,
        }
    delta = (merged["strategy"] - merged["reference"]).to_numpy(dtype=float)
    lo, hi = _bootstrap_ci(delta)
    return {
        "strategy": strategy,
        "reference": reference,
        "budget": float(budget),
        "n_blocks": int(len(delta)),
        "delta": float(np.mean(delta)),
        "ci_low": lo,
        "ci_high": hi,
        "wins": int((delta > 0).sum()),
    }


def _targets_to_threshold(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, seed, strategy), group in frame.groupby(["fold", "acquisition_seed", "strategy"]):
        if strategy == "full_atlas":
            continue
        reference = frame[
            (frame["fold"] == fold)
            & (frame["acquisition_seed"] == seed)
            & (frame["strategy"] == "full_atlas")
        ]["pair_spearman"]
        if reference.empty or not np.isfinite(reference.iloc[0]) or reference.iloc[0] <= 0:
            continue
        threshold = 0.9 * float(reference.iloc[0])
        eligible = group[group["pair_spearman"] >= threshold].sort_values("budget")
        rows.append(
            {
                "fold": int(fold),
                "acquisition_seed": int(seed),
                "strategy": strategy,
                "targets_to_90pct_full": float(eligible["n_targets"].iloc[0])
                if not eligible.empty
                else float("inf"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reports/phase7")
    parser.add_argument("--primary-budget", type=float, default=0.40)
    parser.add_argument("--gate", type=float, default=0.02)
    args = parser.parse_args()
    root = resolve_path(args.root, PROJECT_ROOT)
    metrics = pd.read_csv(root / "metrics.csv")
    curves = (
        metrics.groupby(["strategy", "budget"])["pair_spearman"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "pair_spearman_mean", "std": "pair_spearman_sd", "count": "n_blocks"})
    )
    curves.to_csv(root / "budget_curves.csv", index=False)
    deltas = pd.DataFrame(
        [
            _paired_delta(metrics, strategy=strategy, budget=float(budget))
            for strategy in sorted(set(metrics["strategy"]) - {"full_atlas"})
            for budget in sorted(metrics["budget"].unique())
        ]
    )
    deltas.to_csv(root / "deltas.csv", index=False)
    threshold = _targets_to_threshold(metrics)
    threshold.to_csv(root / "targets_to_threshold.csv", index=False)

    primary = deltas[
        np.isclose(deltas["budget"], args.primary_budget) & (deltas["strategy"] == "hybrid")
    ]
    h1 = bool(
        not primary.empty
        and float(primary["delta"].iloc[0]) >= args.gate
        and float(primary["ci_low"].iloc[0]) > 0
    )
    primary_table = metrics[
        np.isclose(metrics["budget"], args.primary_budget) & (metrics["strategy"] != "full_atlas")
    ]
    strategy_summary = (
        primary_table.groupby("strategy")[["pair_spearman", "profile_normalized", "coverage_calibrated"]]
        .mean()
        .reset_index()
        .sort_values("pair_spearman", ascending=False)
        .to_dict(orient="records")
    )
    summary = {
        "primary_budget": args.primary_budget,
        "gate_absolute_delta": args.gate,
        "primary_hybrid_vs_random": primary.to_dict(orient="records"),
        "h1_supported": h1,
        "strategy_summary_primary": strategy_summary,
        "n_metric_rows": int(len(metrics)),
        "full_atlas_mean_pair_spearman": float(
            metrics[metrics["strategy"] == "full_atlas"]["pair_spearman"].mean()
        ),
        "targets_to_90pct_full_median": {
            str(k): (None if math.isinf(float(v)) else float(v))
            for k, v in (
                threshold.groupby("strategy")["targets_to_90pct_full"].median().to_dict()
                if not threshold.empty
                else {}
            ).items()
        },
    }
    edge_path = root / "connectivity" / "edge_metrics.csv"
    if edge_path.exists():
        edge = pd.read_csv(edge_path)
        by_strategy = edge.set_index("strategy")
        h2 = float("nan")
        if {"discrepancy", "uncertainty"}.issubset(by_strategy.index):
            h2 = float(
                by_strategy.loc["discrepancy", "auroc_anatomy_unexplained"]
                - by_strategy.loc["uncertainty", "auroc_anatomy_unexplained"]
            )
        h4 = float("nan")
        if {"hybrid", "random"}.issubset(by_strategy.index):
            h4 = float(
                by_strategy.loc["hybrid", "auroc_direct_edge"]
                - by_strategy.loc["random", "auroc_direct_edge"]
            )
        summary["edge_metrics"] = edge.to_dict(orient="records")
        summary["h2_discrepancy_minus_uncertainty_auroc"] = h2
        summary["h2_supported"] = bool(h2 >= 0.05)
        summary["h4_hybrid_minus_random_edge_auroc"] = h4
        summary["h4_supported"] = bool(h4 > 0)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
