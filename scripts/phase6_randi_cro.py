"""Run the Phase 6 Causal Response Operator benchmark.

Examples
--------
Smoke test on a small animal subset::

    python scripts/phase6_randi_cro.py --max-rows 20000 --splits unseen_targets --seeds 17

Full preregistered run::

    python scripts/phase6_randi_cro.py --config configs/phase6/cro.yaml
"""

from __future__ import annotations

import argparse
import json

import yaml

from neurobranch.benchmark.randi_response import BenchmarkConfig, write_phase6_cro
from neurobranch.config import PROJECT_ROOT, resolve_path


def _config(path: str | None, args: argparse.Namespace) -> BenchmarkConfig:
    raw = {}
    if path:
        p = resolve_path(path, PROJECT_ROOT)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if args.splits:
        raw["splits"] = tuple(args.splits.split(","))
    if args.seeds:
        raw["seeds"] = tuple(int(x) for x in args.seeds.split(","))
    if args.max_rows is not None:
        raw["max_rows"] = args.max_rows
    if args.output_dir:
        raw["output_dir"] = args.output_dir
    if args.epochs is not None:
        raw["epochs"] = args.epochs
    if args.n_boot is not None:
        raw["n_boot"] = args.n_boot
    known = {f for f in BenchmarkConfig.__dataclass_fields__}
    clean = {k: v for k, v in raw.items() if k in known}
    for key in ("splits", "seeds"):
        if key in clean:
            clean[key] = tuple(clean[key])
    return BenchmarkConfig(**clean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase6/cro.yaml")
    parser.add_argument("--splits", default=None, help="comma-separated splits")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    args = parser.parse_args()
    cfg = _config(args.config, args)
    summary = write_phase6_cro(config=cfg)
    print(json.dumps(summary["pooled"], indent=2)[:8000])


if __name__ == "__main__":
    main()
