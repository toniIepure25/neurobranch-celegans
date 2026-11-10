"""Compare the neural forward/reverse readout with independent reversal events.

The Atanas release stores ``reversal_events`` as frame intervals.  This script
aligns those intervals to the smoothed/clamped trace window and reports whether
the paper's neural readout (and the clamped simulation) tracks an external
behavioural annotation.

This is a behavioural sanity check, not a causal test.  Event intervals are
interpreted as 0-based ``[start, end)``; the trace window starts at raw frame 5
because the notebook removes one edge frame and then takes ``tstart=5``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from neurobranch.bundlenet_sim.clamped import ClampedParameters, simulate_clamped
from neurobranch.bundlenet_sim.premotor import load_premotor
from neurobranch.bundlenet_sim.recording import (
    INTRINSIC_NEURONS,
    build_clamped_recording,
    load_atanas_recording,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT
    / "references"
    / "external"
    / "Celegans_premotor"
    / "data"
    / "Atanas_data"
)
TRACE_OFFSET = 5
FORWARD = ("AVBL", "AVBR", "RIBL", "RIBR", "RID")
REVERSE = ("AVAL", "AVAR", "RIML", "RIMR", "AVEL", "AVER")


def event_vector(raw: dict[str, object], n_trace: int) -> np.ndarray:
    """Map 0-based reversal intervals onto the trace window."""
    raw_len = len(raw["trace_array"][0])  # type: ignore[index]
    events = np.zeros(raw_len, dtype=bool)
    for start, end in raw.get("reversal_events", []):  # type: ignore[union-attr]
        start, end = int(start), int(end)
        if not (0 <= start < end <= raw_len):
            raise ValueError(f"invalid reversal interval [{start}, {end}) for {raw_len} frames")
        events[start:end] = True
    if TRACE_OFFSET + n_trace > raw_len:
        raise ValueError("trace window exceeds recording")
    return events[TRACE_OFFSET : TRACE_OFFSET + n_trace]


def _z_score(traces: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    forward = traces[[index[name] for name in FORWARD]].mean(axis=0)
    reverse = traces[[index[name] for name in REVERSE]].mean(axis=0)
    return forward - reverse


def _binary_metrics(events: np.ndarray, labels: np.ndarray, score: np.ndarray) -> dict[str, object]:
    predicted = labels == -1
    tp = int(np.sum(predicted & events))
    fp = int(np.sum(predicted & ~events))
    fn = int(np.sum(~predicted & events))
    tn = int(np.sum(~predicted & ~events))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    observed = (tp + tn) / len(events)
    expected = (
        ((tp + fp) / len(events)) * ((tp + fn) / len(events))
        + ((fn + tn) / len(events)) * ((fp + tn) / len(events))
    )
    kappa = (observed - expected) / max(1 - expected, 1e-12)
    return {
        "n_frames": int(len(events)),
        "event_prevalence": float(np.mean(events)),
        "predicted_reverse_rate": float(np.mean(predicted)),
        "binary_agreement": float(observed),
        "reversal_precision": float(precision),
        "reversal_recall": float(recall),
        "reversal_f1": float(f1),
        "cohen_kappa": float(kappa),
        "reversal_auc": float(roc_auc_score(events, score)),
        "contingency": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def build_recording(recording_id: str) -> dict[str, object]:
    path = DATA / f"{recording_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    model = load_premotor()
    recording = build_clamped_recording(
        model, load_atanas_recording(path), expected_core=None
    )
    result = simulate_clamped(model, recording, ClampedParameters())
    events = event_vector(raw, len(recording.behavior))
    truth_z = _z_score(recording.traces, recording.core_names)
    sim_z = _z_score(result.intrinsic, INTRINSIC_NEURONS)
    return {
        "recording": recording_id,
        "n_core": recording.n_core,
        "n_frames": len(events),
        "event_interval": "[start, end) 0-based",
        "trace_offset_frames": TRACE_OFFSET,
        "readout": _binary_metrics(events, recording.behavior, -truth_z),
        "model": _binary_metrics(events, result.behavior_sim, -sim_z),
        "readout_label_counts": {
            str(k): int(v) for k, v in zip((-1, 0, 1), np.bincount(recording.behavior + 1, minlength=3))
        },
        "model_label_counts": {
            str(k): int(v) for k, v in zip((-1, 0, 1), np.bincount(result.behavior_sim + 1, minlength=3))
        },
    }


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
        default=ROOT / "reports" / "vienna_workbench" / "tier3_behavior_validation.json",
    )
    args = parser.parse_args()

    records = [build_recording(recording_id) for recording_id in args.recordings]
    readout = [record["readout"] for record in records]
    model = [record["model"] for record in records]
    payload = {
        "format": "vienna-tier3-behavior-validation-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "claim_level": "external behavioural sanity check; not causal",
        "alignment": {
            "trace_offset_frames": TRACE_OFFSET,
            "event_interval": "[start, end) 0-based",
            "note": "Offset follows the notebook's one-frame crop plus tstart=5.",
        },
        "aggregate": {
            "n_recordings": len(records),
            "mean_readout_auc": float(np.mean([r["reversal_auc"] for r in readout])),
            "mean_readout_agreement": float(np.mean([r["binary_agreement"] for r in readout])),
            "mean_readout_precision": float(np.mean([r["reversal_precision"] for r in readout])),
            "mean_readout_recall": float(np.mean([r["reversal_recall"] for r in readout])),
            "mean_model_auc": float(np.mean([r["reversal_auc"] for r in model])),
            "mean_model_agreement": float(np.mean([r["binary_agreement"] for r in model])),
        },
        "recordings": records,
        "caveats": [
            "Reversal intervals are an external annotation, but the [start, end) convention and frame offset are inferred from the release, not supplied as metadata.",
            "This is binary reversal detection, not the full eight-behaviour classification.",
            "The model comparison is a simulation-to-annotation check, not an intervention or causal effect.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
