"""Evaluate simulated trajectory branching against observed held-out reversals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig
from neurobranch.simulator.branching import branch_summary, observed_local, observed_null
from neurobranch.simulator.latent import PCAEncoder
from neurobranch.simulator.train import checkpoint_root, load_datasets, load_simulator_for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.load(args.config)
    root = checkpoint_root(cfg)
    reports = args.reports.resolve()
    selection = json.loads((reports / "selection.json").read_text(encoding="utf-8"))
    selected = str(selection["selected_kind"])
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    nulls: list[dict[str, Any]] = []

    for data in load_datasets(cfg):
        encoder = PCAEncoder.fit(data, n_components=3)
        sim = load_simulator_for(data, selected, args.seed, root=root)
        persistence = load_simulator_for(data, "persistence", args.seed, root=root)
        worm_rows, summary = branch_summary(
            data,
            encoder,
            sim,
            persistence,
            horizon=args.horizon,
            n_rollouts=args.rollouts,
            noise_std=args.noise_std,
            seed=args.seed,
        )
        rows.extend(worm_rows)
        summaries.append(summary)
        _, candidates = observed_local(data, encoder, horizon=args.horizon)
        null = observed_null(
            data,
            encoder,
            candidates,
            horizon=args.horizon,
            n_permutations=args.permutations,
            seed=args.seed,
        )
        nulls.append(
            {
                "worm_id": int(data.worm_id),
                "candidate_frames": int(len(candidates)),
                **{k: v for k, v in null.items() if isinstance(v, (int, float, str))},
            }
        )

    events = pd.DataFrame(rows)
    events.to_csv(reports / "branching_events.csv", index=False)

    for summary, null in zip(summaries, nulls):
        obs_entropies = [
            r["observed_entropy"] for r in rows if r["worm_id"] == summary["worm_id"]
        ]
        lo, hi = np.quantile(obs_entropies, [0.025, 0.975]) if len(obs_entropies) > 1 else (obs_entropies[0], obs_entropies[0])
        summary["observed_entropy_q025"] = float(lo)
        summary["observed_entropy_q975"] = float(hi)
        summary["entropy_inside_observed"] = bool(lo <= summary["mean_entropy_simulated"] <= hi)
        summary["beats_persistence"] = bool(summary["mean_tv_sim"] < summary["mean_tv_persistence"] - 0.02)
        summary["beats_marginal"] = bool(summary["mean_tv_sim"] < summary["mean_tv_marginal"] - 0.02)
        summary["passed"] = bool(
            summary["beats_persistence"] and summary["beats_marginal"] and summary["entropy_inside_observed"]
        )
        summary["observed_null"] = null

    n_passed = sum(bool(s["passed"]) for s in summaries)
    gate = {
        "status": "pass" if n_passed >= 3 else "fail",
        "criterion": (
            "mean TV(sim, observed) at least 0.02 lower than both constant-state persistence "
            "and the outcome-marginal null, and simulated entropy inside the observed 95% "
            "event range, in >=3/5 worms"
        ),
        "n_worms_passed": int(n_passed),
        "worms": summaries,
        "interpretation": (
            "A pass would mean the simulator reproduces conditional outcome distributions "
            "better than null baselines. It would not establish a bifurcation or a causal "
            "branching mechanism."
        ),
        "observed_null_interpretation": (
            "Block-permutation nulls test observed local entropy against geometry-unrelated "
            "outcomes; a significant deficit is not evidence for branching."
        ),
    }
    (reports / "branching_summary.json").write_text(
        json.dumps({"schema_version": 1, "gate_3": gate}, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(events)} event rows; gate_3={gate['status']} ({n_passed}/5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
