"""Build the Tier-3 clamped-core validation report.

Usage:
    python scripts/build_tier3_validation.py

The report is deliberately explicit about what it does *not* show.  The
published equations and core counts are reproduced, but the recovered weights
are an independent reconstruction; pointwise agreement with the paper's figure
is not claimed without a reference Julia run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neurobranch.bundlenet_sim.clamped import (
    ClampedParameters,
    negative_controls,
    simulate_clamped,
    validation_record,
)
from neurobranch.bundlenet_sim.premotor import load_premotor
from neurobranch.bundlenet_sim.recording import build_clamped_recording, load_atanas_recording

ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT
    / "references"
    / "external"
    / "Celegans_premotor"
    / "data"
    / "Atanas_data"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["2023-01-23-15", "2023-01-10-07", "2023-01-09-15"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "vienna_workbench" / "tier3_clamped_validation.json",
    )
    parser.add_argument("--beta", type=float, default=10.0)
    args = parser.parse_args()

    model = load_premotor()
    records: list[dict[str, object]] = []
    control_records: dict[str, object] = {}
    main_result = None
    for recording_id in args.recordings:
        path = DATA / f"{recording_id}.json"
        recording = build_clamped_recording(
            model, load_atanas_recording(path), expected_core=None
        )
        result = simulate_clamped(model, recording, ClampedParameters(beta=args.beta))
        if main_result is None:
            main_result = result
        record = validation_record(result)
        record.update(
            {
                "n_core": recording.n_core,
                "n_intrinsic": recording.n_intrinsic,
                "n_signal": recording.n_signal,
                "replacement_map": recording.replacement_map,
                "n_frames": recording.n_frames,
            }
        )
        records.append(record)
        if recording_id == args.recordings[0]:
            controls = negative_controls(model, recording, ClampedParameters(beta=args.beta))
            control_records = {
                name: validation_record(value) for name, value in controls.items()
            }
            for name, value in controls.items():
                delta = np.abs(value.intrinsic - main_result.intrinsic)
                control_records[name]["delta_vs_main"] = {
                    "mean_abs_intrinsic": float(np.mean(delta)),
                    "max_abs_intrinsic": float(np.max(delta)),
                }

    agreements = [float(r["metrics"]["state_agreement"]) for r in records]
    js = [float(r["metrics"]["behavior_distribution_js"]) for r in records]
    max_abs = [float(r["metrics"]["max_abs"]) for r in records]
    payload = {
        "format": "vienna-tier3-clamped-validation-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "claim_level": "published-parameter replay/sandbox; not a digital twin",
        "reference_status": (
            "Julia reference notebook was not run in this environment; the recovered "
            "A/d/W and neuron order are validated by support checks and contract tests, "
            "but pointwise fidelity to the paper figure is not established."
        ),
        "recordings": records,
        "controls": control_records,
        "aggregate": {
            "n_recordings": len(records),
            "mean_state_agreement": float(np.mean(agreements)) if agreements else float("nan"),
            "mean_behavior_distribution_js": float(np.mean(js)) if js else float("nan"),
            "max_abs": float(np.max(max_abs)) if max_abs else float("nan"),
        },
        "caveats": [
            "Only the canonical 2023-01-23-15 recording has the published 84/15/69 counts.",
            "Other recordings are shorter cores; they are reported with their actual counts.",
            "The recovered gap-junction matrix W is not part of the public JLD2 export.",
            "A low state agreement is evidence against a pointwise replay claim, not evidence for mechanism.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
