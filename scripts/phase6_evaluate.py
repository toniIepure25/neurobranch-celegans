"""Phase 6 exploratory test evaluation under the frozen autonomous contract.

The test block was already inspected in Phase 5.  Every row emitted here is
therefore labelled exploratory replication, never confirmatory evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, sha256_file
from neurobranch.dynamics.models import BehaviorProbe, PersistenceModel
from neurobranch.preprocessing import split_starts
from neurobranch.simulator import NeuralSimulator
from neurobranch.simulator.metrics import one_step_metrics, rollout_metrics
from neurobranch.simulator.train import checkpoint_root, load_datasets, load_simulator_for

HORIZONS = (20, 100)
H20_STARTS = 50
H100_STARTS = 20
H200_STARTS = 10
SEEDS = (17, 23, 31)


def _starts(data, split: str, *, horizon: int, n: int, history: int = 3) -> np.ndarray:
    starts = split_starts(data, split, history=history, horizon=horizon)
    if len(starts) > n:
        idx = np.linspace(0, len(starts) - 1, n).round().astype(int)
        starts = starts[np.unique(idx)]
    return np.asarray(starts, dtype=np.int64)


def _persistence(data) -> NeuralSimulator:
    train = split_starts(data, "train", history=1, horizon=1)
    probe = BehaviorProbe(seed=17).fit(data.X[train], data.y_macro[train])
    return NeuralSimulator(PersistenceModel(n_neurons=data.n_neurons, probe=probe), data)


def _load_phase6(data, family: str, root: Path) -> NeuralSimulator:
    path = root / "phase6" / family / f"worm{data.worm_id}.pt"
    return NeuralSimulator.from_checkpoint(path, data)


def _models(data, family: str, root: Path):
    if family == "persistence":
        yield "persistence", _persistence(data)
    elif family == "phase5_mlp":
        yield "phase5_mlp", load_simulator_for(data, "mlp_residual", 17, root=root)
    elif family == "phase5_var":
        yield "phase5_var", load_simulator_for(data, "var", 17, root=root)
    else:
        sim = _load_phase6(data, family, root)
        yield family, sim


def _row_base(family: str, data, horizon: int, seed: int, n_starts: int) -> dict[str, Any]:
    return {
        "family": family,
        "worm_id": int(data.worm_id),
        "horizon": horizon,
        "seed": seed,
        "n_starts": n_starts,
        "evidence_tier": "exploratory",
        "test_block_reused": True,
    }


def evaluate(datasets, root: Path, reports: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = ("persistence", "phase5_mlp", "phase5_var", "var", "stochastic_var", "switching_var")
    summary_rows: list[dict[str, Any]] = []
    per_start_rows: list[dict[str, Any]] = []
    h200_rows: list[dict[str, Any]] = []
    for data in datasets:
        for horizon in HORIZONS:
            n = H20_STARTS if horizon == 20 else H100_STARTS
            starts = _starts(data, "test", horizon=horizon, n=n)
            for family in families:
                for name, sim in _models(data, family, root):
                    seeds = SEEDS if name in {"stochastic_var", "switching_var"} else (17,)
                    for seed in seeds:
                        summary, per_start = rollout_metrics(
                            sim,
                            starts,
                            horizon=horizon,
                            noise_std=0.0,
                            base_seed=seed * 1000,
                        )
                        row = _row_base(name, data, horizon, seed, len(starts))
                        row.update(summary)
                        row["one_step_r2"] = one_step_metrics(sim)["r2"]
                        row["native_stochastic"] = bool(
                            getattr(sim.model, "native_stochastic", False)
                        )
                        row["checkpoint"] = sim.checkpoint
                        summary_rows.append(row)
                        for r in per_start:
                            rr = _row_base(name, data, horizon, seed, len(starts))
                            rr.update(r)
                            per_start_rows.append(rr)
                        print(
                            f"{name} worm {data.worm_id} H{horizon} seed {seed}: "
                            f"skill={summary['skill_vs_persistence']:.4f}",
                            flush=True,
                        )
        # Long-horizon stability is diagnostic, not a separate fit.
        starts = _starts(data, "test", horizon=200, n=H200_STARTS)
        for family in ("var", "stochastic_var", "switching_var"):
            for name, sim in _models(data, family, root):
                seed = 17
                max_abs, norm_ratio, finite = [], [], []
                for i, t in enumerate(starts):
                    hist = sim.initial_history(
                        int(t), history=int(getattr(sim.model, "history_len", 1))
                    )
                    rollout = sim.rollout(
                        None,
                        200,
                        initial_history=hist,
                        seed=seed * 1000 + i,
                    )
                    finite.append(bool(np.isfinite(rollout.states).all()))
                    max_abs.append(float(np.nanmax(np.abs(rollout.states))))
                    denom = float(np.sqrt(np.mean(hist[-1] ** 2)))
                    norm_ratio.append(float(np.sqrt(np.mean(rollout.states[-1] ** 2))) / max(denom, 1e-9))
                h200_rows.append(
                    {
                        **_row_base(name, data, 200, seed, len(starts)),
                        "finite": bool(np.all(finite)),
                        "max_abs": float(np.nanmax(max_abs)) if max_abs else float("nan"),
                        "final_norm_ratio": float(np.nanmax(norm_ratio)) if norm_ratio else float("nan"),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(per_start_rows), pd.DataFrame(h200_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/phase6")
    args = parser.parse_args(argv)
    cfg = ExperimentConfig.load(args.config)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(cfg)
    manifest = {
        "phase": 6,
        "evidence_tier": "exploratory",
        "test_block_reused": True,
        "config_sha256": sha256_file(args.config),
        "data_sha256": sha256_file(cfg.data.path),
        "horizons": list(HORIZONS),
        "starts": {
            str(d.worm_id): {
                "h20": _starts(d, "test", horizon=20, n=H20_STARTS).tolist(),
                "h100": _starts(d, "test", horizon=100, n=H100_STARTS).tolist(),
                "h200": _starts(d, "test", horizon=200, n=H200_STARTS).tolist(),
            }
            for d in datasets
        },
    }
    (reports / "test_start_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    summary, per_start, h200 = evaluate(datasets, checkpoint_root(cfg), reports)
    summary.to_csv(reports / "test_summary.csv", index=False)
    per_start.to_csv(reports / "test_per_start.csv", index=False)
    h200.to_csv(reports / "long_run_stability.csv", index=False)
    by = (
        summary.groupby(["family", "horizon"], as_index=False)
        .agg(
            mean_skill=("skill_vs_persistence", "mean"),
            median_skill=("skill_vs_persistence", "median"),
            mean_energy=("energy_distance", "mean"),
            mean_ood=("ood_fraction", "mean"),
            finite_fraction=("finite_fraction", "mean"),
        )
        .to_dict(orient="records")
    )
    payload = {
        "evidence_tier": "exploratory",
        "test_block_reused": True,
        "n_summary_rows": len(summary),
        "by_family_horizon": by,
        "start_manifest": "test_start_manifest.json",
        "warning": "Block 2 was inspected in Phase 5; these are not confirmatory results.",
    }
    (reports / "test_overview.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(by, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
