"""GENFAIL Stage B: forecastability map with simple baselines."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim
from neurobranch.diagnostics.forecastability import (
    evaluate_models,
    noise_sensitivity,
    period_stability,
)

OUT = PROJECT_ROOT / "reports/simulator/failure_diagnostics/forecastability"
HORIZONS = (1, 3, 5, 10, 20, 50, 100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", default="1,3,5,10,20,50,100")
    args = parser.parse_args(argv)
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, datasets, _ = load_nostim()
    summary_rows = []
    neuron_rows = []
    growth_rows = []
    period_rows = []
    noise_rows = []
    for data in datasets:
        print(f"forecastability worm {data.worm_id}", flush=True)
        summary, neurons, growth = evaluate_models(data, horizons=horizons, history=5)
        summary_rows.extend(summary)
        neuron_rows.extend(neurons)
        growth_rows.extend(growth)
        period_rows.extend(period_stability(data, horizon=20, history=5))
        noise_rows.extend(noise_sensitivity(data, horizon=20, history=5))

    summary_df = pd.DataFrame(summary_rows)
    neuron_df = pd.DataFrame(neuron_rows)
    growth_df = pd.DataFrame(growth_rows)
    period_df = pd.DataFrame(period_rows)
    noise_df = pd.DataFrame(noise_rows)
    summary_df.to_csv(OUT / "forecast_summary.csv", index=False)
    neuron_df.to_csv(OUT / "forecast_by_neuron.csv", index=False)
    growth_df.to_csv(OUT / "error_growth.csv", index=False)
    period_df.to_csv(OUT / "period_stability.csv", index=False)
    noise_df.to_csv(OUT / "noise_sensitivity.csv", index=False)

    agg = (
        summary_df.groupby(["model", "horizon"], as_index=False)
        .agg(
            mean_skill=("skill_vs_persistence", "mean"),
            median_skill=("skill_vs_persistence", "median"),
            min_skill=("skill_vs_persistence", "min"),
            max_skill=("skill_vs_persistence", "max"),
            n_sessions=("worm_id", "nunique"),
            mean_r2=("r2", "mean"),
        )
        .sort_values(["horizon", "mean_skill"], ascending=[True, False])
    )
    agg.to_csv(OUT / "forecast_aggregate.csv", index=False)

    pop = summary_df[summary_df["model"].isin(["neuron_ar5", "population_var"])].pivot_table(
        index=["worm_id", "horizon"], columns="model", values="mse"
    )
    pop = pop.reset_index()
    pop["population_minus_independent_mse"] = pop["population_var"] - pop["neuron_ar5"]
    pop["population_beats_independent"] = pop["population_minus_independent_mse"] < 0
    pop.to_csv(OUT / "population_vs_independent.csv", index=False)

    metadata = {
        "stage": "B_forecastability",
        "evidence_tier": "exploratory",
        "source_sha256": datasets[0].recording.source_sha256,
        "config": str(cfg.data.path),
        "horizons_frames": list(horizons),
        "sampling": {str(d.worm_id): float(d.recording.fps) for d in datasets},
        "history_frames": 5,
        "test_starts": "all starts wholly inside the chronological test mask",
        "bootstrap": "moving-block, block=20, 400 draws, descriptive",
        "models": [
            "persistence",
            "train_mean",
            "local_trend",
            "neuron_ar5",
            "population_var",
            "latent_lds",
        ],
        "selection": "none on test; prespecified model settings",
        "git": git_state(),
    }
    (OUT / "forecastability_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
