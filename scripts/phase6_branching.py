"""Phase 6 stochastic branching against empirical, marginal and Markov references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig
from neurobranch.simulator import NeuralSimulator
from neurobranch.simulator.branching import branch_summary
from neurobranch.simulator.latent import PCAEncoder
from neurobranch.simulator.train import checkpoint_root, load_datasets

FAMILIES = ("stochastic_var", "switching_var")


def _persistence(data) -> NeuralSimulator:
    from neurobranch.dynamics.models import BehaviorProbe, PersistenceModel
    from neurobranch.preprocessing import split_starts

    train = split_starts(data, "train", history=1, horizon=1)
    probe = BehaviorProbe(seed=17).fit(data.X[train], data.y_macro[train])
    return NeuralSimulator(PersistenceModel(n_neurons=data.n_neurons, probe=probe), data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/phase6")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--rollouts", type=int, default=120)
    args = parser.parse_args(argv)
    cfg = ExperimentConfig.load(args.config)
    root = checkpoint_root(cfg)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(cfg)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for data in datasets:
        encoder = PCAEncoder.fit(data, n_components=20)
        persistence = _persistence(data)
        for family in args.families:
            sim = NeuralSimulator.from_checkpoint(
                root / "phase6" / family / f"worm{data.worm_id}.pt", data
            )
            rows, summary = branch_summary(
                data,
                encoder,
                sim,
                persistence,
                horizon=20,
                n_rollouts=args.rollouts,
                noise_std=0.0,
                seed=17,
            )
            for r in rows:
                r["family"] = family
            summary["family"] = family
            all_rows.extend(rows)
            summaries.append(summary)
            print(
                f"{family} worm {data.worm_id}: states={summary['n_states']} "
                f"TVsim={summary['mean_tv_sim']:.3f} TVmarkov={summary['mean_tv_markov']:.3f} "
                f"TVmarg={summary['mean_tv_marginal']:.3f}",
                flush=True,
            )
    rows_frame = pd.DataFrame(all_rows)
    summary_frame = pd.DataFrame(summaries)
    rows_frame.to_csv(reports / "branching_events.csv", index=False)
    summary_frame.to_csv(reports / "branching_summary.csv", index=False)
    aggregate = (
        summary_frame.groupby("family", as_index=False)
        .agg(
            n_worms=("worm_id", "nunique"),
            n_states=("n_states", "sum"),
            mean_tv_sim=("mean_tv_sim", "mean"),
            mean_tv_persistence=("mean_tv_persistence", "mean"),
            mean_tv_marginal=("mean_tv_marginal", "mean"),
            mean_tv_markov=("mean_tv_markov", "mean"),
            mean_entropy_observed=("mean_entropy_observed", "mean"),
            mean_entropy_simulated=("mean_entropy_simulated", "mean"),
            mean_ood=("mean_ood_fraction", "mean"),
        )
        .to_dict(orient="records")
    )
    payload = {
        "evidence_tier": "exploratory",
        "test_block_reused": True,
        "horizon": 20,
        "n_rollouts_per_state": args.rollouts,
        "families": list(args.families),
        "aggregate": aggregate,
        "note": "Empirical reference is held-out kNN conditional distribution; no shuffled-null entropy claim.",
    }
    (reports / "branching_overview.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
