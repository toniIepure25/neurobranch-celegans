"""Validate a newly released worm and optionally run the held-out Tier-1 benchmark.

This is the pre-registered ingestion gate for a sixth worm.  It never invents
labels or fills missing frames: it fails loudly on shape, label or identifier
mismatches, then optionally writes a benchmark report that includes the new
worm in the leave-one-worm-out folds.

Expected BunDLe-Net file:
    bundlenet_consistent_embedding_worm_<id>.npz
    keys: x (T, 3) float, b (T,) integer in 0..7

Optional NC-MCM file:
    nc_mcm_model_worm_<id+1>.npz
    keys: arr_0..arr_5 as documented in neurobranch.bundlenet_sim.ncmcm

Example:
    python scripts/validate_new_worm.py --worm-id 5 --run-benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neurobranch.bundlenet_sim.benchmark import benchmark_models, save_benchmark  # noqa: E402
from neurobranch.bundlenet_sim.dataset import (  # noqa: E402
    BEHAVIOR_NAMES,
    available_worm_ids,
    load_trajectories,
)
from neurobranch.bundlenet_sim.ncmcm import load_ncmcm  # noqa: E402


def validate(worm_id: int, data_dir: Path, ncmcm_dir: Path | None) -> dict[str, object]:
    ids = available_worm_ids(data_dir)
    if worm_id not in ids:
        raise SystemExit(
            f"worm {worm_id} not found in {data_dir}; discovered ids={list(ids)}. "
            "Check the filename pattern and rerun."
        )
    trajectories = load_trajectories(data_dir)
    trajectory = next(t for t in trajectories if t.worm_id == worm_id)
    counts = np.bincount(trajectory.behavior, minlength=len(BEHAVIOR_NAMES))
    result: dict[str, object] = {
        "worm_id": worm_id,
        "n_frames": int(len(trajectory.latent)),
        "latent_shape": list(trajectory.latent.shape),
        "behavior_counts": {name: int(counts[i]) for i, name in enumerate(BEHAVIOR_NAMES)},
        "all_behaviors_present": bool(np.all(counts > 0)),
        "source": trajectory.source,
        "n_discovered_worms": len(ids),
        "discovered_worm_ids": list(ids),
    }
    if ncmcm_dir is not None:
        nc = load_ncmcm(worm_id, ncmcm_dir)
        result["ncmcm"] = {
            "n_frames": int(len(nc.behavior)),
            "behavior_names": list(nc.behavior_names),
            "behavior_counts": np.bincount(nc.behavior, minlength=len(BEHAVIOR_NAMES)).tolist(),
        }
        if len(nc.behavior) != len(trajectory.latent):
            result["ncmcm"]["warning"] = (
                "frame counts differ; establish the interval/offset convention before alignment"
            )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worm-id", type=int, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "bundlenet_sim",
    )
    parser.add_argument(
        "--ncmcm-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "ncmcm",
    )
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--stride", type=int, default=75)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "vienna_workbench" / "tier1_benchmark_with_new_worm.json",
    )
    args = parser.parse_args(argv)

    summary = validate(args.worm_id, args.data_dir, args.ncmcm_dir)
    print(json.dumps(summary, indent=2))
    if args.run_benchmark:
        trajectories = load_trajectories(args.data_dir)
        if len(trajectories) < 6:
            raise SystemExit(
                f"benchmark needs at least 6 worms for a meaningful new held-out fold; "
                f"found {len(trajectories)}"
            )
        result = benchmark_models(
            trajectories,
            horizon=args.horizon,
            stride=args.stride,
            seeds=args.seeds,
        )
        save_benchmark(result, args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
