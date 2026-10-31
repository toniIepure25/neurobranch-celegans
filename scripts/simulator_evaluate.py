"""Evaluate free-running simulator models and the frozen Gate 2 rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from neurobranch.config import PROJECT_ROOT, ExperimentConfig
from neurobranch.preprocessing import split_starts
from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.metrics import one_step_metrics, rollout_metrics
from neurobranch.simulator.train import (
    DEFAULT_KINDS,
    DEFAULT_SEEDS,
    checkpoint_root,
    load_datasets,
    load_simulator_for,
)

REPORTS = PROJECT_ROOT / "reports" / "simulator"


def _paired_predictions(sim: NeuralSimulator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = int(getattr(sim.model, "history_len", 1))
    starts = split_starts(sim.data, "test", history=history, horizon=1)
    H = np.stack([sim.data.X[t - history + 1 : t + 1] for t in starts])
    pred = np.asarray(sim.model.predict_next(H), dtype=np.float64)
    truth = sim.data.X[starts + 1]
    persistence = sim.data.X[starts]
    return pred, truth, persistence


def _moving_block_indices(n: int, *, block: int = 10, rng: np.random.Generator) -> np.ndarray:
    if n <= block:
        return rng.choice(n, size=n, replace=True)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def _paired_ci(
    pred: np.ndarray,
    truth: np.ndarray,
    persistence: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = _moving_block_indices(len(truth), rng=rng)
        y = truth[idx]
        p = pred[idx]
        q = persistence[idx]
        if np.var(y) <= 1e-12:
            continue
        deltas.append(float(r2_score(y.ravel(), p.ravel()) - r2_score(y.ravel(), q.ravel())))
    point = float(r2_score(truth.ravel(), pred.ravel()) - r2_score(truth.ravel(), persistence.ravel()))
    if not deltas:
        return {"delta_r2": point, "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "delta_r2": point,
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }


def _composite(dist: dict[str, Any]) -> float:
    keys = ("mean_abs_error", "mean_sd_error", "autocorr_error", "cross_corr_error")
    vals = [float(dist.get(k, np.nan)) for k in keys]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cfg = ExperimentConfig.load(args.config)
    root = checkpoint_root(cfg)
    datasets = load_datasets(cfg)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    one_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selection = json.loads((reports / "selection.json").read_text(encoding="utf-8"))
    selected_kind = str(selection["selected_kind"])
    kinds = tuple(args.kinds) if args.kinds else tuple(DEFAULT_KINDS)
    seeds = tuple(args.seeds) if args.seeds else tuple(DEFAULT_SEEDS)

    for kind in kinds:
        for data in datasets:
            for seed in seeds:
                try:
                    sim = load_simulator_for(data, kind, seed, root=root)
                except (FileNotFoundError, ValueError) as exc:
                    rows.append({"kind": kind, "worm_id": data.worm_id, "seed": seed, "error": str(exc)})
                    continue
                starts = sim.test_starts(horizon=args.horizon)[: args.starts]
                if len(starts) == 0:
                    rows.append({"kind": kind, "worm_id": data.worm_id, "seed": seed, "error": "no starts"})
                    continue
                summary, per_start = rollout_metrics(
                    sim,
                    starts,
                    horizon=args.horizon,
                    noise_std=args.noise_std,
                    base_seed=1000 + seed,
                )
                one = one_step_metrics(sim)
                row = {
                    "kind": kind,
                    "worm_id": int(data.worm_id),
                    "seed": seed,
                    "history": int(getattr(sim.model, "history_len", 1)),
                    **summary,
                    "one_step_r2": one["r2"],
                    "one_step_mse": one["mse"],
                    "persistence_one_step_r2": one["persistence_r2"],
                    "delta_r2_vs_persistence": one["delta_r2_vs_persistence"],
                    "composite_distribution_error": _composite(summary),
                }
                rows.append(row)
                one_rows.append({"kind": kind, "worm_id": data.worm_id, "seed": seed, **one})
                if kind == selected_kind:
                    pred, truth, persistence = _paired_predictions(sim)
                    ci = _paired_ci(pred, truth, persistence, seed=seed)
                    selected_rows.append(
                        {
                            "kind": kind,
                            "worm_id": int(data.worm_id),
                            "seed": seed,
                            "delta_r2": ci["delta_r2"],
                            "ci_low": ci["ci_low"],
                            "ci_high": ci["ci_high"],
                            "mse_h5": summary["mse_h5"],
                            "persistence_h5": float(
                                np.mean([r["persistence_h5"] for r in per_start if r.get("finite")])
                            ),
                            "composite": _composite(summary),
                            "n_starts": len(starts),
                            "finite_fraction": summary["finite_fraction"],
                            "ood_fraction": summary["ood_fraction"],
                        }
                    )

    model_df = pd.DataFrame(rows)
    one_df = pd.DataFrame(one_rows)
    selected_df = pd.DataFrame(selected_rows)
    model_df.to_csv(reports / "rollout_metrics.csv", index=False)
    one_df.to_csv(reports / "one_step_metrics.csv", index=False)
    selected_df.to_csv(reports / "selected_model_test.csv", index=False)

    # Gate 1: selected model, seed 17, every worm, 20 steps, finite.
    g1_rows = model_df[
        (model_df["kind"] == selected_kind)
        & (model_df["seed"] == 17)
        & (model_df.get("finite_fraction", 0) >= 0.95)
    ]
    g1 = bool(len(g1_rows) == len(datasets))

    # Gate 2 uses worm-level mean across seeds. The bootstrap CI is computed
    # for each seed; the gate requires the seed-17 CI and mean effect.
    gate2_worms = []
    for data in datasets:
        sub = selected_df[selected_df["worm_id"] == data.worm_id]
        if sub.empty:
            gate2_worms.append({"worm_id": data.worm_id, "passed": False, "reason": "missing"})
            continue
        mean_delta = float(sub["delta_r2"].mean())
        mean_mse5 = float(sub["mse_h5"].mean())
        mean_pers5 = float(sub["persistence_h5"].mean())
        seed17 = sub[sub["seed"] == 17].iloc[0]
        passed = bool(
            mean_delta >= 0.02
            and float(seed17["ci_low"]) > 0
            and mean_mse5 <= mean_pers5
        )
        gate2_worms.append(
            {
                "worm_id": int(data.worm_id),
                "mean_delta_r2": mean_delta,
                "seed17_ci_low": float(seed17["ci_low"]),
                "seed17_ci_high": float(seed17["ci_high"]),
                "mean_mse5": mean_mse5,
                "mean_persistence_h5": mean_pers5,
                "passed": passed,
            }
        )
    n_pass = sum(bool(w["passed"]) for w in gate2_worms)
    g2 = bool(n_pass >= 4)

    payload = {
        "schema_version": 1,
        "selected_kind": selected_kind,
        "selection": {
            "path": "selection.json",
            "criterion": selection.get("criterion"),
            "selected_kind": selection.get("selected_kind"),
            "selected_means": selection.get("selected_means"),
            "means": selection.get("means"),
            "test_block_used_for_selection": selection.get("test_block_used_for_selection"),
        },
        "config": args.config.name,
        "horizon": int(args.horizon),
        "starts": int(args.starts),
        "noise_std": float(args.noise_std),
        "seeds": list(seeds),
        "n_models": int(len(model_df)),
        "gate_1": {
            "status": "pass" if g1 else "fail",
            "criterion": "selected model, seed 17, 5/5 worms, finite fraction >= 0.95",
            "n_worms": len(g1_rows),
        },
        "gate_2": {
            "status": "pass" if g2 else "fail",
            "criterion": "mean delta R2 >= 0.02, seed-17 block-bootstrap CI > 0, mean h5 MSE <= persistence, in >=4/5 worms",
            "n_worms_passed": int(n_pass),
            "worms": gate2_worms,
        },
        "gate_6": {
            "status": "not_evaluable",
            "reason": "Kato WT_NoStim has no matched named-neuron intervention.",
        },
    }
    (reports / "summary.json").write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=float))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=REPORTS)
    parser.add_argument("--kinds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.0)
    args = parser.parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
