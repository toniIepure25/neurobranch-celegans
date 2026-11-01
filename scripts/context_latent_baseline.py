"""Stage B: temporal-holdout latent LDS baseline and stability comparison.

Selection uses validation rollouts only.  The temporal test block is touched
once for the globally selected configuration and for the pre-specified
baselines.  Frozen Phase 5/6 checkpoints were trained on a different (blocked)
split and are reported descriptively, not as matched retrainings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, git_state
from neurobranch.dynamics.context_models import StableLatentLDS
from neurobranch.dynamics.stable_var import StableVARModel
from neurobranch.dynamics.training import load_worm_data
from neurobranch.preprocessing import split_starts
from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.metrics import one_step_metrics, rollout_metrics

LATENT_DIMS = (5, 10, 15)
RIDGES = (1e-2, 1e-1)
CAPS = (0.995, 0.95)
VAR_LAGS = (1, 2)
VAR_RIDGES = (1e-2, 1.0)


def _bootstrap_ci(rows: list[dict[str, Any]], *, n: int = 1000, seed: int = 17) -> dict[str, float]:
    usable = [r for r in rows if r.get("finite") and np.isfinite(r.get("mse", np.nan))]
    if len(usable) < 5:
        return {"lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    n_rows = len(usable)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        mse = np.array([usable[i]["mse"] for i in idx], dtype=float)
        pers = np.array([usable[i]["persistence_mse"] for i in idx], dtype=float)
        denom = pers.mean()
        vals.append(float(1 - mse.mean() / denom) if denom > 0 else np.nan)
    vals = np.asarray(vals)
    return {
        "lo": float(np.nanpercentile(vals, 2.5)),
        "hi": float(np.nanpercentile(vals, 97.5)),
    }


def _eval(sim: NeuralSimulator, split: str, horizon: int) -> tuple[dict[str, Any], list]:
    starts = (
        sim.validation_starts(horizon=horizon)
        if split == "val"
        else sim.test_starts(horizon=horizon)
    )
    return rollout_metrics(sim, starts=starts, horizon=horizon)


def _row(
    data,
    model,
    split: str,
    horizon: int,
    *,
    model_name: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sim = NeuralSimulator(model, data)
    summary, rows = _eval(sim, split, horizon)
    ci = _bootstrap_ci(rows)
    return (
        {
            "model": model_name,
            "worm_id": int(data.worm_id),
            "split": split,
            "horizon": int(horizon),
            "n_starts": summary.get("n_starts"),
            "mse": summary.get("mse"),
            "persistence_mse": summary.get("persistence_mse"),
            "skill_vs_persistence": summary.get("skill_vs_persistence"),
            "skill_ci_low": ci["lo"],
            "skill_ci_high": ci["hi"],
            "r2": summary.get("r2"),
            "energy_distance": summary.get("energy_distance"),
            "autocorr_error": summary.get("autocorr_error"),
            "variance_ratio": summary.get("variance_ratio"),
            "ood_fraction_q99": summary.get("ood_fraction"),
            "support_q90_fraction": summary.get("support_q90_fraction"),
            "support_q95_fraction": summary.get("support_q95_fraction"),
            "support_q99_fraction": summary.get("support_q99_fraction"),
            **{f"cfg_{k}": v for k, v in config.items()},
        },
        rows,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/forward_stress.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/context_dynamics")
    parser.add_argument("--rollouts-dir", type=Path, default=PROJECT_ROOT / "reports/simulator/context_dynamics/trajectories")
    args = parser.parse_args(argv)
    cfg = ExperimentConfig.load(args.config)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    traj = Path(args.rollouts_dir)
    traj.mkdir(parents=True, exist_ok=True)
    datasets = load_worm_data(cfg)

    validation_rows: list[dict[str, Any]] = []
    fitted: dict[tuple[int, ...], Any] = {}
    for data in datasets:
        train_starts = split_starts(data, "train", history=1, horizon=1)
        for dim in LATENT_DIMS:
            for ridge in RIDGES:
                for cap in CAPS:
                    model = StableLatentLDS.fit(
                        data.X,
                        data.y_macro,
                        train_starts,
                        train_mask=data.masks["train"],
                        latent_dim=dim,
                        ridge=ridge,
                        spectral_cap=cap,
                    )
                    key = (int(data.worm_id), dim, int(round(np.log10(ridge) * 1000)), int(round(cap * 10000)))
                    fitted[key] = model
                    summary, rows = _eval(NeuralSimulator(model, data), "val", 20)
                    validation_rows.append(
                        {
                            "worm_id": int(data.worm_id),
                            "latent_dim": dim,
                            "ridge": ridge,
                            "spectral_cap": cap,
                            "n_starts": summary["n_starts"],
                            "skill_vs_persistence": summary["skill_vs_persistence"],
                            "mse": summary["mse"],
                            "persistence_mse": summary["persistence_mse"],
                            "r2": summary["r2"],
                            "explained_variance_ratio": model.diagnostics["explained_variance_ratio"],
                        }
                    )
        print(f"latent validation worm {data.worm_id} done", flush=True)

    val_frame = pd.DataFrame(validation_rows)
    grouped = (
        val_frame.groupby(["latent_dim", "ridge", "spectral_cap"], as_index=False)
        .agg(
            median_skill=("skill_vs_persistence", "median"),
            mean_skill=("skill_vs_persistence", "mean"),
            n_worms=("worm_id", "nunique"),
        )
        .sort_values("median_skill", ascending=False)
    )
    selected = grouped.iloc[0].to_dict()
    selected_cfg = {
        "latent_dim": int(selected["latent_dim"]),
        "ridge": float(selected["ridge"]),
        "spectral_cap": float(selected["spectral_cap"]),
    }
    selected_key = (
        selected_cfg["latent_dim"],
        int(round(np.log10(selected_cfg["ridge"]) * 1000)),
        int(round(selected_cfg["spectral_cap"] * 10000)),
    )

    free_rows: list[dict[str, Any]] = []
    per_start: list[dict[str, Any]] = []
    for data in datasets:
        model = fitted[(int(data.worm_id), *selected_key)]
        for horizon in (20, 100):
            row, rows = _row(
                data,
                model,
                "test",
                horizon,
                model_name="latent_lds",
                config=selected_cfg,
            )
            free_rows.append(row)
            for r in rows:
                per_start.append({**r, "model": "latent_lds", "worm_id": int(data.worm_id)})
        # Save the selected deterministic trajectory from the first eligible start.
        starts = NeuralSimulator(model, data).test_starts(horizon=100)
        if len(starts):
            start = int(starts[len(starts) // 2])
            sim = NeuralSimulator(model, data)
            rollout = sim.rollout(
                None,
                100,
                initial_history=sim.initial_history(start),
                seed=17,
            )
            np.savez_compressed(
                traj / f"latent_lds_worm{data.worm_id}.npz",
                simulated=rollout.states,
                observed=data.X[start : start + 101],
                frame=np.array([start]),
                model=np.array(["latent_lds"]),
            )
        one = one_step_metrics(NeuralSimulator(model, data))
        free_rows.append(
            {
                "model": "latent_lds_one_step",
                "worm_id": int(data.worm_id),
                "split": "test",
                "horizon": 1,
                "n_starts": one["n_starts"],
                "mse": one["mse"],
                "persistence_mse": one["persistence_mse"],
                "skill_vs_persistence": 1 - one["mse"] / one["persistence_mse"]
                if one["persistence_mse"] > 0
                else float("nan"),
                "r2": one["r2"],
                **{f"cfg_{k}": v for k, v in selected_cfg.items()},
            }
        )
        print(f"latent test worm {data.worm_id} done", flush=True)

    # Corrected stable VAR, fitted on the same temporal training block.
    var_rows: list[dict[str, Any]] = []
    var_val_rows: list[dict[str, Any]] = []
    var_fitted: dict[tuple[int, int, float], Any] = {}
    for data in datasets:
        for lag in VAR_LAGS:
            starts = split_starts(data, "train", history=lag, horizon=1)
            for ridge in VAR_RIDGES:
                model = StableVARModel.fit(
                    data.X,
                    data.y_macro,
                    starts,
                    history=lag,
                    ridge=ridge,
                    spectral_cap=0.995,
                )
                key = (int(data.worm_id), lag, ridge)
                var_fitted[key] = model
                summary, _ = _eval(NeuralSimulator(model, data), "val", 20)
                var_val_rows.append(
                    {
                        "worm_id": int(data.worm_id),
                        "lag": lag,
                        "ridge": ridge,
                        "skill_vs_persistence": summary["skill_vs_persistence"],
                        "realized_spectral_radius": model.realized_spectral_radius,
                    }
                )
    var_val = pd.DataFrame(var_val_rows)
    var_grouped = (
        var_val.groupby(["lag", "ridge"], as_index=False)
        .agg(median_skill=("skill_vs_persistence", "median"))
        .sort_values("median_skill", ascending=False)
    )
    var_selected = var_grouped.iloc[0].to_dict()
    var_cfg = {"lag": int(var_selected["lag"]), "ridge": float(var_selected["ridge"])}
    for data in datasets:
        model = var_fitted[(int(data.worm_id), var_cfg["lag"], var_cfg["ridge"])]
        for horizon in (20, 100):
            row, rows = _row(
                data,
                model,
                "test",
                horizon,
                model_name="stable_var",
                config={"history": var_cfg["lag"], "ridge": var_cfg["ridge"]},
            )
            row["realized_spectral_radius"] = model.realized_spectral_radius
            var_rows.append(row)
            for r in rows:
                per_start.append({**r, "model": "stable_var", "worm_id": int(data.worm_id)})

    # Frozen baselines: descriptive only (different training split).
    frozen_rows: list[dict[str, Any]] = []
    frozen_specs = [
        ("phase5_var_frozen", "artifacts/runs/blocked_suite/var__worm{w}__seed17/model.pt"),
        ("latent_mlp_frozen", "artifacts/runs/blocked_suite/latent_mlp__worm{w}__seed17/model.pt"),
        ("latent_knn_frozen", "artifacts/runs/blocked_suite/latent_knn__worm{w}__seed17/model.pt"),
    ]
    for data in datasets:
        for name, template in frozen_specs:
            path = PROJECT_ROOT / template.format(w=data.worm_id)
            if not path.exists():
                continue
            try:
                sim = NeuralSimulator.from_checkpoint(path, data)
                for horizon in (20, 100):
                    row, _ = _row(
                        data,
                        sim.model,
                        "test",
                        horizon,
                        model_name=name,
                        config={"checkpoint": path.name, "training_split": "blocked_phase6"},
                    )
                    frozen_rows.append(row)
            except Exception as exc:  # noqa: BLE001 - report, do not silently skip
                frozen_rows.append(
                    {
                        "model": name,
                        "worm_id": int(data.worm_id),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    all_rows = free_rows + var_rows + frozen_rows
    free_frame = pd.DataFrame(all_rows)
    free_frame.to_csv(reports / "free_running_latent.csv", index=False)
    pd.DataFrame(per_start).to_csv(reports / "free_running_latent_per_start.csv", index=False)
    val_frame.to_csv(reports / "latent_dimension_validation.csv", index=False)
    var_val.to_csv(reports / "stable_var_validation.csv", index=False)
    grouped.to_csv(reports / "latent_model_selection.csv", index=False)

    payload = {
        "schema_version": 1,
        "stage": "B_latent_lds",
        "evidence_tier": "exploratory_temporal_holdout",
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "selected_latent": selected_cfg,
        "selected_stable_var": var_cfg,
        "selection_rule": "highest median validation H20 skill vs persistence across the five temporal holdouts; test not used",
        "test_block": "final 1/6 of each NoStim recording; same recordings as Phase 6, new chronological split; NOT animal-independent",
        "frozen_baselines": "trained on the Phase 6 blocked split; descriptive only",
        "git": git_state(),
        "n_models_latent": len(fitted),
        "n_models_var": len(var_fitted),
    }
    (reports / "model_selection.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
