"""Stage C: compact compositional latent operator model with causal coefficients.

This is not a reproduction of the published dLDS implementation.  It uses the
published mathematical form as inspiration, is fitted independently, and keeps
the distinction between retrospective operator diagnostics and a causal forecast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, git_state
from neurobranch.dynamics.context_models import CompositionalLatentLDS
from neurobranch.dynamics.training import load_worm_data
from neurobranch.preprocessing import split_starts
from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.metrics import one_step_metrics, rollout_metrics

OPERATOR_COUNTS = (1, 2, 3)


def _eval(data, model, split: str, horizon: int):
    sim = NeuralSimulator(model, data)
    starts = (
        sim.validation_starts(horizon=horizon)
        if split == "val"
        else sim.test_starts(horizon=horizon)
    )
    if horizon == 100 and len(starts) > 80:
        idx = np.linspace(0, len(starts) - 1, 80).round().astype(int)
        starts = starts[np.unique(idx)]
    return rollout_metrics(sim, starts=starts, horizon=horizon)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/forward_stress.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/context_dynamics")
    args = parser.parse_args(argv)
    cfg = ExperimentConfig.load(args.config)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    selected = json.loads((reports / "model_selection.json").read_text(encoding="utf-8"))["selected_latent"]
    datasets = load_worm_data(cfg)

    val_rows: list[dict[str, Any]] = []
    models: dict[tuple[int, int], Any] = {}
    for data in datasets:
        starts = split_starts(data, "train", history=1, horizon=1)
        for n_ops in OPERATOR_COUNTS:
            model = CompositionalLatentLDS.fit(
                data.X,
                data.y_macro,
                starts,
                train_mask=data.masks["train"],
                latent_dim=int(selected["latent_dim"]),
                ridge=float(selected["ridge"]),
                spectral_cap=float(selected["spectral_cap"]),
                n_operators=int(n_ops),
                alternations=5,
            )
            models[(int(data.worm_id), int(n_ops))] = model
            summary, _ = _eval(data, model, "val", 20)
            val_rows.append(
                {
                    "worm_id": int(data.worm_id),
                    "n_operators": int(n_ops),
                    "skill_vs_persistence": summary["skill_vs_persistence"],
                    "mse": summary["mse"],
                    "persistence_mse": summary["persistence_mse"],
                    "r2": summary["r2"],
                    "mean_coefficient_entropy": model.coefficient_diagnostics["mean_coefficient_entropy"],
                    "coefficient_policy_rmse": model.coefficient_diagnostics["coefficient_policy_rmse"],
                    "max_operator_radius": model.coefficient_diagnostics["max_operator_radius"],
                }
            )
        print(f"operator validation worm {data.worm_id} done", flush=True)

    val = pd.DataFrame(val_rows)
    selection = (
        val.groupby("n_operators", as_index=False)
        .agg(median_skill=("skill_vs_persistence", "median"), n_worms=("worm_id", "nunique"))
        .sort_values("median_skill", ascending=False)
    )
    chosen = int(selection.iloc[0]["n_operators"])
    test_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for data in datasets:
        model = models[(int(data.worm_id), chosen)]
        for horizon in (20, 100):
            summary, rows = _eval(data, model, "test", horizon)
            test_rows.append(
                {
                    "model": f"compositional_m{chosen}",
                    "worm_id": int(data.worm_id),
                    "horizon": horizon,
                    "n_starts": summary["n_starts"],
                    "mse": summary["mse"],
                    "persistence_mse": summary["persistence_mse"],
                    "skill_vs_persistence": summary["skill_vs_persistence"],
                    "r2": summary["r2"],
                    "energy_distance": summary["energy_distance"],
                    "support_q90_fraction": summary["support_q90_fraction"],
                    "support_q95_fraction": summary["support_q95_fraction"],
                    "support_q99_fraction": summary["support_q99_fraction"],
                }
            )
        one = one_step_metrics(NeuralSimulator(model, data))
        test_rows.append(
            {
                "model": f"compositional_m{chosen}_one_step",
                "worm_id": int(data.worm_id),
                "horizon": 1,
                "n_starts": one["n_starts"],
                "mse": one["mse"],
                "persistence_mse": one["persistence_mse"],
                "skill_vs_persistence": 1 - one["mse"] / one["persistence_mse"]
                if one["persistence_mse"] > 0
                else float("nan"),
                "r2": one["r2"],
            }
        )
        # Causal coefficient trajectory on the test block.  The policy sees only
        # the current encoded state (and current known input); it never sees a
        # future observation or an oracle coefficient path.
        encoded = model.encode(data.X)
        for t in range(1, data.n_frames):
            if not data.masks["test"][t]:
                continue
            c = model.coefficients_for(encoded[t - 1], np.zeros(model.input_dim))
            coefficient_rows.append(
                {
                    "worm_id": int(data.worm_id),
                    "frame": int(t),
                    "coefficient_mode": "causal_encoded_state",
                    **{f"c{m + 1}": float(v) for m, v in enumerate(c)},
                }
            )
        print(f"operator test worm {data.worm_id} done", flush=True)

    val.to_csv(reports / "operator_validation.csv", index=False)
    pd.DataFrame(test_rows).to_csv(reports / "operator_test.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(reports / "operator_coefficient_trajectories.csv", index=False)
    diag = {
        "schema_version": 1,
        "stage": "C_compositional_operators",
        "evidence_tier": "exploratory_temporal_holdout",
        "selected_latent": selected,
        "selected_n_operators": chosen,
        "selection_rule": "median validation H20 skill across worms; test not used",
        "evaluation_note": "H20 uses every held-out start; H100 uses up to 80 evenly spaced starts",
        "reference": {
            "paper_doi": "10.1038/s42003-025-08599-3",
            "code_zenodo": "10.5281/zenodo.15866270",
            "relation": "independent implementation of the published operator form; not a reproduction and not biologically identified",
        },
        "operator_diagnostics": {
            f"worm{data.worm_id}": models[(int(data.worm_id), chosen)].coefficient_diagnostics
            for data in datasets
        },
        "git": git_state(),
        "warning": "Coefficients/operators are model-internal, not synaptic or circuit mechanisms.",
    }
    (reports / "operator_summary.json").write_text(
        json.dumps(diag, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected_n_operators": chosen, "selection": selection.to_dict("records")}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
