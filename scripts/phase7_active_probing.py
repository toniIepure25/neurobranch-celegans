"""CLI for the Phase 7 pseudo-prospective active acquisition benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml

from neurobranch.active.simulate import ActiveConfig, run_active_benchmark
from neurobranch.benchmark.randi_response import load_trials
from neurobranch.config import PROJECT_ROOT, resolve_path


def _load_config(path: str | None) -> ActiveConfig:
    if path is None:
        return ActiveConfig()
    p = resolve_path(path, PROJECT_ROOT)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    allowed = set(ActiveConfig.__dataclass_fields__)
    values = {k: v for k, v in raw.items() if k in allowed}
    for key in ("budgets", "strategies", "acquisition_seeds", "model_seeds", "final_model_seeds"):
        if key in values and values[key] is not None:
            values[key] = tuple(values[key])
    return ActiveConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase7/active.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="comma-separated acquisition seeds")
    parser.add_argument("--model-seeds", default=None, help="comma-separated model seeds")
    parser.add_argument("--budgets", default=None, help="comma-separated fractions")
    parser.add_argument("--strategies", default=None, help="comma-separated strategies")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    updates = {}
    if args.output_dir:
        updates["output_dir"] = args.output_dir
    if args.folds is not None:
        updates["n_folds"] = args.folds
    if args.epochs is not None:
        updates["epochs"] = args.epochs
    if args.seeds:
        updates["acquisition_seeds"] = tuple(int(x) for x in args.seeds.split(","))
    if args.model_seeds:
        updates["model_seeds"] = tuple(int(x) for x in args.model_seeds.split(","))
    if args.budgets:
        updates["budgets"] = tuple(float(x) for x in args.budgets.split(","))
    if args.strategies:
        updates["strategies"] = tuple(args.strategies.split(","))
    if args.smoke:
        updates.update(
            {
                "output_dir": args.output_dir or "reports/phase7/smoke",
                "n_folds": 5,
                "acquisition_seeds": (101,),
                "model_seeds": (17,),
                "budgets": (0.20, 0.40),
                "strategies": ("random", "uncertainty", "hybrid"),
                "epochs": 2,
                "smoke": True,
            }
        )
    cfg = replace(cfg, **updates)
    trials = load_trials(cfg.trials_path)
    if cfg.smoke:
        # Keep the smoke affordable while retaining the real target/split logic.
        animals = sorted(trials["dataset_id"].astype(str).unique())[:24]
        trials = trials[trials["dataset_id"].astype(str).isin(animals)].reset_index(drop=True)
    result = run_active_benchmark(trials, config=cfg)
    out = Path(cfg.output_dir)
    summary = {
        "config": cfg.__dict__,
        "audit": result["audit"],
        "n_metric_rows": int(len(result["metrics"])),
        "metrics_tail": result["metrics"].tail(12).to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary["audit"], indent=2))
    print(result["metrics"].tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
