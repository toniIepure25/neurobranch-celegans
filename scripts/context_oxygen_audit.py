"""Audit the measured Kato WT_Stim oxygen schedule and split power.

The oxygen switches are an environmental sensory input, not targeted neuronal
stimulation.  Seven files are sessions, not seven independent animals.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, git_state, sha256_file
from neurobranch.data.external import (
    build_kato_stim_context_worm_data,
    load_kato_stim,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_path = Path(os.environ.get("TEMP", ".")) / "neurobranch_external" / "WT_Stim.mat"
    parser.add_argument("--stim", type=Path, default=default_path)
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/context_dynamics")
    args = parser.parse_args(argv)
    if not args.stim.exists():
        raise SystemExit(f"Kato WT_Stim not found: {args.stim}")
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    recordings = load_kato_stim(args.stim, verify=True)
    session_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for rec in recordings:
        oxygen = rec.oxygen_percent
        if oxygen is None:
            raise RuntimeError(f"{rec.session_id}: no oxygen schedule")
        data = build_kato_stim_context_worm_data(rec)
        changes = np.flatnonzero(np.diff(oxygen) != 0) + 1
        for idx in changes:
            timing_rows.append(
                {
                    "session_id": rec.session_id,
                    "frame": int(idx),
                    "time_seconds": float(rec.time_seconds[idx]),
                    "from_percent": float(oxygen[idx - 1]),
                    "to_percent": float(oxygen[idx]),
                }
            )
        split_counts = {}
        for name, mask in data.masks.items():
            split_counts[name] = {
                "n_frames": int(mask.sum()),
                "first_frame": int(np.flatnonzero(mask)[0]),
                "last_frame": int(np.flatnonzero(mask)[-1]),
                "n_oxygen_transitions_inside": int(
                    sum(bool(mask[idx] and mask[idx - 1]) for idx in changes)
                ),
            }
        session_rows.append(
            {
                "session_id": rec.session_id,
                "worm_index": int(rec.worm_index),
                "dataset_name": rec.dataset_name,
                "n_frames": int(rec.n_frames),
                "duration_seconds": float(rec.time_seconds[-1]),
                "fps": float(rec.fps),
                "n_predictor_traces": int(data.n_neurons),
                "n_behavior_traces_excluded": int(len(data.excluded_neurons)),
                "n_oxygen_transitions": int(len(changes)),
                "oxygen_levels": sorted({float(v) for v in np.unique(oxygen)}),
                "first_transition_seconds": float(rec.time_seconds[changes[0]]),
                "manifest_sha256": sha256_file(args.stim),
                "splits": split_counts,
                "warning": "session, not an independently identified animal; oxygen is a controlled environmental input",
            }
        )
        print(f"audited {rec.session_id}", flush=True)
    session = pd.DataFrame(session_rows)
    timing = pd.DataFrame(timing_rows)
    session.to_csv(reports / "oxygen_session_audit.csv", index=False)
    timing.to_csv(reports / "oxygen_transition_times.csv", index=False)
    payload = {
        "schema_version": 1,
        "stage": "D0_oxygen_input_audit",
        "source": {
            "file": args.stim.name,
            "sha256": sha256_file(args.stim),
            "path_recorded_as": "external local path; not copied into Git",
            "dataset": "Kato et al. (2015) WT_Stim",
        },
        "n_sessions": int(len(session)),
        "n_unique_sessions": int(session["session_id"].nunique()),
        "n_transitions_per_session": sorted(set(session["n_oxygen_transitions"].astype(int))),
        "transition_seconds": sorted(set(round(v, 6) for v in timing["time_seconds"])),
        "oxygen_levels": sorted(set(v for row in session["oxygen_levels"] for v in row)),
        "first_segment_constant_percent": 21.0,
        "interpretation": (
            "Oxygen is a measured environmental sensory input, not optogenetic or "
            "targeted neuronal stimulation. Seven sessions do not establish animal independence."
        ),
        "power_warning": (
            "Transitions are repeated in a fixed 30 s schedule; effective independent "
            "transition events are few. All input-model inferential results are low power."
        ),
        "git": git_state(),
    }
    (reports / "oxygen_input_audit.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
