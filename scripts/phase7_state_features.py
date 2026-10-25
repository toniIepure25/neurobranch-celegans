"""Build the Phase 7 pre-stimulus state cache from raw Randi recordings."""

from __future__ import annotations

import argparse
import json

import numpy as np

from neurobranch.active.state_features import build_state_cache, save_state_cache
from neurobranch.benchmark.randi_response import load_trials
from neurobranch.config import PROJECT_ROOT, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default="reports/phase3/randi_trial_table_phase3.csv.gz")
    parser.add_argument("--data-dir", default="data/raw/randi2023/exported_data")
    parser.add_argument("--output", default="reports/phase7/state/event_state.npz")
    parser.add_argument("--pre-frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    trials = load_trials(args.trials)
    if args.limit:
        animals = sorted(trials["dataset_id"].astype(str).unique())[: args.limit]
        trials = trials[trials["dataset_id"].astype(str).isin(animals)].reset_index(drop=True)
    cache = build_state_cache(
        trials,
        data_dir=resolve_path(args.data_dir, PROJECT_ROOT),
        pre_frames=args.pre_frames,
    )
    out = resolve_path(args.output, PROJECT_ROOT)
    save_state_cache(cache, out)
    summary = {
        "n_events": cache.n_events,
        "n_nodes": len(cache.node_names),
        "n_masked_finite": int(cache.mask.sum()),
        "fraction_finite": float(cache.mask.mean()),
        "output": str(out),
        "sha256": __import__("hashlib").sha256(out.read_bytes()).hexdigest(),
    }
    (out.parent / "state_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
