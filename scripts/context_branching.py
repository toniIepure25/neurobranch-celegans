"""Stage F: branching from fitted process uncertainty, not arbitrary noise.

The latent model's stochastic ensemble samples its fitted one-step residual
covariance.  This is model uncertainty, not proof of biological bifurcation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, git_state
from neurobranch.dynamics.context_models import StableLatentLDS
from neurobranch.dynamics.models import BehaviorProbe, PersistenceModel
from neurobranch.dynamics.training import load_worm_data
from neurobranch.preprocessing import split_starts
from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.branching import branch_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/forward_stress.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/context_dynamics")
    parser.add_argument("--n-rollouts", type=int, default=60)
    args = parser.parse_args(argv)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    selected = json.loads((reports / "model_selection.json").read_text(encoding="utf-8"))["selected_latent"]
    cfg = ExperimentConfig.load(args.config)
    datasets = load_worm_data(cfg)
    rows, aggregates = [], []
    for data in datasets:
        model = StableLatentLDS.fit(
            data.X,
            data.y_macro,
            split_starts(data, "train", history=1, horizon=1),
            train_mask=data.masks["train"],
            latent_dim=int(selected["latent_dim"]),
            ridge=float(selected["ridge"]),
            spectral_cap=float(selected["spectral_cap"]),
        )
        probe = BehaviorProbe(seed=17)
        probe.fit(data.X[data.masks["train"]], data.y_macro[data.masks["train"]])
        persistence = PersistenceModel(n_neurons=data.n_neurons, probe=probe)
        sim = NeuralSimulator(model, data)
        psim = NeuralSimulator(persistence, data)
        try:
            event_rows, summary = branch_summary(
                data,
                model,
                sim,
                psim,
                horizon=20,
                n_rollouts=int(args.n_rollouts),
                noise_std=0.0,
                stochastic=True,
                seed=17,
            )
        except ValueError as exc:
            aggregates.append({"worm_id": int(data.worm_id), "error": str(exc)})
            continue
        rows.extend(event_rows)
        summary["process_cov_trace"] = float(np.trace(model.process_cov))
        summary["process_cov_max_eigenvalue"] = float(np.max(np.linalg.eigvalsh(model.process_cov)))
        summary["noise_source"] = "fitted latent process covariance, sampled with the model seed"
        aggregates.append(summary)
        print(f"branching worm {data.worm_id} done", flush=True)
    pd.DataFrame(rows).to_csv(reports / "branching_events.csv", index=False)
    pd.DataFrame(aggregates).to_csv(reports / "branching_aggregate.csv", index=False)
    payload = {
        "schema_version": 1,
        "stage": "F_branching",
        "evidence_tier": "exploratory_temporal_holdout",
        "model": "selected latent LDS",
        "ensemble": "stochastic draws from fitted process covariance; no arbitrary neural noise",
        "comparators": ["persistence", "first-order Markov", "marginal first-exit", "observed local kNN outcome distribution"],
        "interpretation_warning": (
            "Agreement or disagreement with the observed local outcome distribution is a model diagnostic. "
            "It is not evidence that the living network contains a bifurcation or that the model has identified one."
        ),
        "n_rollouts_per_state": int(args.n_rollouts),
        "git": git_state(),
    }
    (reports / "branching_summary.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(aggregates).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
