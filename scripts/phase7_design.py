"""Phase 7 experiment designer.

Ranks target-level hypotheses using predicted effect, interval width, animal
coverage and an anatomy-unexplained discrepancy term.  This is a hypothesis
generator, not an interventional effect estimate.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from neurobranch.causal.design import WARNING
from neurobranch.config import PROJECT_ROOT, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reports/phase7")
    parser.add_argument("--strategy", default="hybrid")
    parser.add_argument("--top-targets", type=int, default=15)
    parser.add_argument("--top-pairs", type=int, default=8)
    args = parser.parse_args()
    root = resolve_path(args.root, PROJECT_ROOT)
    pred_path = root / "predictions.parquet"
    if not pred_path.exists():
        raise SystemExit("primary predictions missing")
    frame = pd.read_parquet(pred_path)
    if "strategy" not in frame.columns:
        frame["strategy"] = frame["model"]
    frame = frame[frame["strategy"] == args.strategy].copy()
    if frame.empty:
        raise SystemExit(f"no predictions for strategy {args.strategy!r}")
    frame["abs_effect"] = frame["y_pred"].abs()
    frame["width"] = (frame["upper"] - frame["lower"]).clip(lower=0)
    target = frame.groupby("target", as_index=False).agg(
        mean_abs_effect=("abs_effect", "mean"),
        mean_width=("width", "mean"),
        n_events=("y_true", "size"),
        n_animals=("dataset_id", "nunique"),
        n_responders=("neuron", "nunique"),
    )
    discrepancy_path = root / "discrepancy_pairs.csv"
    if discrepancy_path.exists():
        disc = pd.read_csv(discrepancy_path)
        counts = disc.groupby("target").size()
        target["discrepancy_pairs"] = target["target"].map(counts).fillna(0).astype(int)
    else:
        target["discrepancy_pairs"] = 0
    max_disc = max(float(target["discrepancy_pairs"].max()), 1.0)
    target["discrepancy_norm"] = target["discrepancy_pairs"] / max_disc
    target["coverage_factor"] = 1.0 / np.sqrt(target["n_animals"].clip(lower=1))
    target["priority_score"] = (
        target["mean_abs_effect"]
        * np.sqrt(target["mean_width"].clip(lower=0))
        * target["coverage_factor"]
        * (1.0 + 0.5 * target["discrepancy_norm"])
    )
    target = target.replace([np.inf, -np.inf], np.nan).dropna(subset=["priority_score"])
    target = target.sort_values("priority_score", ascending=False).head(args.top_targets)
    target["warning"] = WARNING
    target["recommended_experiment"] = (
        "stimulate target in >=3 new animals; measure top predicted responders and "
        "at least 2 predicted-null responders; test the anatomy-unexplained candidates"
    )
    out = root / "design"
    out.mkdir(parents=True, exist_ok=True)
    target.to_csv(out / "target_candidates.csv", index=False)

    pair_rows = []
    for target_name in target["target"]:
        group = frame[frame["target"] == target_name].copy()
        group = group.groupby(["target", "neuron"], as_index=False).agg(
            predicted_effect=("y_pred", "mean"),
            observed_effect=("y_true", "mean"),
            mean_width=("width", "mean"),
            n_events=("y_true", "size"),
        )
        group["abs_effect"] = group["predicted_effect"].abs()
        group = group.sort_values("abs_effect", ascending=False).head(args.top_pairs)
        group["warning"] = WARNING
        pair_rows.append(group)
    pairs = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    pairs.to_csv(out / "pair_candidates.csv", index=False)
    summary = {
        "warning": WARNING,
        "strategy": args.strategy,
        "n_targets_ranked": len(target),
        "n_pair_candidates": len(pairs),
        "score_definition": (
            "mean |predicted effect| * sqrt(mean interval width) / sqrt(n animals) "
            "* (1 + 0.5 * normalised anatomy-unexplained candidate count)"
        ),
        "limitations": [
            "Predictions are observational associations, not causal effects.",
            "The discrepancy term is an acquisition and prioritisation heuristic.",
            "The designer cannot know the true sign of an untested optogenetic effect.",
        ],
        "top_targets": target[
            [
                "target",
                "priority_score",
                "mean_abs_effect",
                "mean_width",
                "n_animals",
                "discrepancy_pairs",
            ]
        ].to_dict(orient="records"),
    }
    (out / "design.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
