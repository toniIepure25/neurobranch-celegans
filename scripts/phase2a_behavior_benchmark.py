"""Phase 2 incremental behaviour benchmark on Kato 2015.

Writes the exact tables used in docs/BEHAVIORAL_GENERALIZATION.md.
Requires the Kato archive at data/raw/kato2015/WT_NoStim.mat.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neurobranch.config import ExperimentConfig
from neurobranch.dynamics.training import load_worm_data
from neurobranch.evaluation.behavior_benchmark import (
    bootstrap_change_accuracy,
    evaluate_worm_behavior,
    incremental_contrasts,
    summarize_behavior,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase2a_scientific_audit" / "behavior"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configs = {
        "blocked": ROOT / "configs" / "experiments" / "blocked_suite.yaml",
        "forward": ROOT / "configs" / "experiments" / "forward_stress.yaml",
    }
    for name, path in configs.items():
        cfg = ExperimentConfig.load(path)
        datasets = load_worm_data(cfg)
        rows = []
        prevalence = {}
        for data in datasets:
            result = evaluate_worm_behavior(data)
            result["protocol"] = name
            rows.append(result)
            prevalence[str(data.worm_id)] = bootstrap_change_accuracy(data, result)
        results = pd.concat(rows, ignore_index=True)
        summary = summarize_behavior(results)
        increments = incremental_contrasts(results)

        results.to_csv(OUT / f"behavior_results_{name}.csv", index=False)
        summary.to_csv(OUT / f"behavior_summary_{name}.csv", index=False)
        increments.drop(columns=["per_worm"]).to_csv(
            OUT / f"behavior_incremental_{name}.csv", index=False
        )
        (OUT / f"change_prevalence_{name}.json").write_text(
            json.dumps(prevalence, indent=2), encoding="utf-8"
        )
        print(f"{name}: {len(results)} rows")
        print(summary[["model", "balanced_accuracy_mean", "macro_f1_mean", "log_loss_mean", "brier_mean"]])
        print(increments[["model", "metric", "mean_delta", "wins", "n_worms"]])


if __name__ == "__main__":
    main()
