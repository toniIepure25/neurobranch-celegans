"""Run the preregistered Phase 8 Adaptive Perturbation Twin benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from neurobranch.config import PROJECT_ROOT, resolve_path
from neurobranch.twin.simulate import TwinRunConfig, run_twin_benchmark


def _load_config(path: Path) -> TwinRunConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    fields = {
        key: payload[key]
        for key in TwinRunConfig.__dataclass_fields__
        if key in payload
    }
    for key in ("model_seeds", "acquisition_seeds", "support_sizes", "strategies"):
        if key in fields:
            fields[key] = tuple(fields[key])
    return TwinRunConfig(**fields)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase8/apt.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(resolve_path(args.config, PROJECT_ROOT))
    if args.output:
        cfg.output_dir = args.output
    if args.folds:
        cfg.n_folds = args.folds
    if args.epochs:
        cfg.epochs = args.epochs
    if args.seeds:
        cfg.model_seeds = tuple(args.seeds)
    if args.n_boot:
        cfg.n_boot = args.n_boot
    if args.smoke:
        cfg.n_folds = 3
        cfg.model_seeds = (17,)
        cfg.acquisition_seeds = (101,)
        cfg.support_sizes = (0, 2, 8)
        cfg.epochs = 2
        cfg.cro_epochs = 1
        cfg.n_boot = 30
        cfg.max_test_animals = 3
        cfg.output_dir = "reports/phase8_smoke_final"
    summary = run_twin_benchmark(cfg)
    print(json.dumps(summary, indent=2))
    print("\nMODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
