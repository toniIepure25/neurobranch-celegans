"""Model-based experiment designer.

The designer ranks *hypotheses worth testing*, not truths.  Every output carries
the explicit warning that virtual perturbations from an observational model are
not causally validated.  Scores combine predicted effect size, predictive
uncertainty and current animal coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED"


def design_experiments(
    predictions: pd.DataFrame,
    *,
    output_dir: str | Path,
    top_targets: int = 12,
    top_pairs_per_target: int = 8,
    model: str = "cro",
) -> dict[str, object]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cro = predictions[predictions["model"] == model].copy()
    if cro.empty:
        raise ValueError(f"no predictions for model {model!r}")
    cro["abs_effect"] = cro["y_pred"].abs()
    if "upper" in cro.columns and "lower" in cro.columns:
        cro["width"] = cro["upper"] - cro["lower"]
    else:
        cro["width"] = np.nan

    target = cro.groupby("target", as_index=False).agg(
        mean_abs_effect=("abs_effect", "mean"),
        mean_width=("width", "mean"),
        n_events=("y_true", "size"),
        n_animals=("dataset_id", "nunique"),
        n_responders=("neuron", "nunique"),
    )
    target["coverage_factor"] = 1.0 / np.sqrt(target["n_animals"].clip(lower=1))
    # Expected value of an experiment: a large predicted effect whose interval is
    # still wide, in a target measured in few animals.  This is a heuristic, not
    # an information-theoretic optimum.
    target["priority_score"] = (
        target["mean_abs_effect"] * np.sqrt(target["mean_width"].clip(lower=0)) * target["coverage_factor"]
    )
    target = target.sort_values("priority_score", ascending=False).head(top_targets)
    target["warning"] = WARNING
    target["recommended_experiment"] = (
        "stimulate target in >=3 new animals; measure the top predicted responders "
        "and at least 2 predicted-null responders"
    )
    target.to_csv(out / "target_candidates.csv", index=False)

    pair_rows = []
    for target_name in target["target"]:
        group = cro[cro["target"] == target_name].copy()
        group = group.groupby(["target", "neuron"], as_index=False).agg(
            predicted_effect=("y_pred", "mean"),
            observed_effect=("y_true", "mean"),
            mean_width=("width", "mean"),
            n_events=("y_true", "size"),
        )
        group["abs_effect"] = group["predicted_effect"].abs()
        group = group.sort_values("abs_effect", ascending=False).head(top_pairs_per_target)
        group["warning"] = WARNING
        pair_rows.append(group)
    pairs = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    pairs.to_csv(out / "pair_candidates.csv", index=False)

    summary = {
        "warning": WARNING,
        "model": model,
        "n_targets_ranked": int(len(target)),
        "n_pair_candidates": int(len(pairs)),
        "score_definition": (
            "mean |predicted effect| * sqrt(mean 90% interval width) / sqrt(n animals); "
            "a heuristic for where a new experiment is both promising and informative"
        ),
        "limitations": [
            "Predictions are observational associations, not causal effects.",
            "Intervals are calibrated to held-out Randi animals, not to new labs or genotypes.",
            "The designer cannot know the true sign of an untested optogenetic effect.",
        ],
        "top_targets": target[
            ["target", "priority_score", "mean_abs_effect", "mean_width", "n_animals"]
        ].to_dict(orient="records"),
    }
    (out / "design.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
