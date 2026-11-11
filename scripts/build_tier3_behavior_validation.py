"""Validate the Tier-3 readout and clamped replay against reversal events.

The report is deliberately baseline-controlled and uncertainty-aware.  It
answers four separate questions:

1. Does the paper's neural readout track an external reversal annotation?
2. Does the clamped replay carry reversal information at zero lag?
3. Is any information retrospective (model response follows the event) or
   prospective (model response precedes the event)?
4. Does the model beat a one-frame persistence baseline?

A positive selected shift means the model response follows the event.  The
leave-one-recording-out protocol chooses that shift on the other recordings
only; the held-out recording is never used to select it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neurobranch.bundlenet_sim.behavior_metrics import (
    DEFAULT_SHIFTS,
    binary_metrics,
    block_bootstrap_ci,
    event_intervals,
    event_level_metrics,
    event_vector,
    lag_auc,
    lag_curve,
    majority_labels,
    persistence_labels,
    select_lag_loo,
    shift_labels,
    shift_score,
)
from neurobranch.bundlenet_sim.clamped import (
    ClampedParameters,
    negative_controls,
    simulate_clamped,
)
from neurobranch.bundlenet_sim.premotor import load_premotor
from neurobranch.bundlenet_sim.recording import (
    INTRINSIC_NEURONS,
    build_clamped_recording,
    load_atanas_recording,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "references" / "external" / "Celegans_premotor" / "data" / "Atanas_data"
TRACE_OFFSET = 5
FORWARD = ("AVBL", "AVBR", "RIBL", "RIBR", "RID")
REVERSE = ("AVAL", "AVAR", "RIML", "RIMR", "AVEL", "AVER")


def _z_score(traces: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    forward = traces[[index[name] for name in FORWARD]].mean(axis=0)
    reverse = traces[[index[name] for name in REVERSE]].mean(axis=0)
    return forward - reverse


def _git_state() -> dict[str, object]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "scripts", "tests"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"git_sha": sha, "dirty": bool(dirty), "dirty_scope": "src scripts tests"}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "dirty": None, "dirty_scope": "src scripts tests"}


def _readout_payload(metrics: dict[str, object], ci: dict[str, object]) -> dict[str, object]:
    return {
        "auc": metrics["auc"],
        "auc_ci": ci["auc_ci"],
        "agreement": metrics["accuracy"],
        "agreement_ci": ci["agreement_ci"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "cohen_kappa": metrics["cohen_kappa"],
        "event_prevalence": metrics["event_prevalence"],
    }


def _model_payload(
    events: np.ndarray,
    labels: np.ndarray,
    score: np.ndarray,
    shift: int,
    dt: float,
) -> dict[str, object]:
    lag0 = binary_metrics(events, labels, score)
    retro_labels = shift_labels(labels, shift)
    retro_score = shift_score(score, shift)
    retrospective = binary_metrics(events, retro_labels, retro_score)
    prospective_shift = -shift if shift != 0 else -8
    prospective = binary_metrics(
        events, shift_labels(labels, prospective_shift), shift_score(score, prospective_shift)
    )
    return {
        "lag0_auc": lag0["auc"],
        "lag0_agreement": lag0["accuracy"],
        "retrospective_auc": retrospective["auc"],
        "retrospective_agreement": retrospective["accuracy"],
        "retrospective_auc_ci": block_bootstrap_ci(events, retro_labels, retro_score)["auc_ci"],
        "selected_shift_frames": int(shift),
        "selected_shift_ms": float(shift * dt * 1000.0),
        "direction": (
            "model_lags_event" if shift > 0 else "model_leads_event" if shift < 0 else "synchronous"
        ),
        "prospective_shift_frames": int(prospective_shift),
        "prospective_auc": prospective["auc"],
        "prospective_agreement": prospective["accuracy"],
    }


def _control_payload(
    controls: dict[str, object], events: np.ndarray, shift: int, dt: float
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name in ("beta0", "zero_signals"):
        result = controls[name]
        score = -_z_score(result.intrinsic, INTRINSIC_NEURONS)
        labels = result.behavior_sim
        lag0 = binary_metrics(events, labels, score)
        aligned = binary_metrics(events, shift_labels(labels, shift), shift_score(score, shift))
        payload[name] = {
            "lag0_auc": lag0["auc"],
            "aligned_auc": aligned["auc"],
            "aligned_agreement": aligned["accuracy"],
            "selected_shift_frames": int(shift),
            "selected_shift_ms": float(shift * dt * 1000.0),
        }
    return payload


def _sensitivity(
    raw: dict[str, object],
    recording_id: str,
    n_trace: int,
    model_labels: np.ndarray,
    model_score: np.ndarray,
) -> dict[str, object]:
    by_offset = []
    for offset in range(3, 8):
        events = event_vector(raw, n_trace, offset=offset)
        metrics = binary_metrics(events, model_labels, model_score)
        by_offset.append({"offset_frames": offset, "model_lag0_auc": metrics["auc"]})
    closed = event_vector(raw, n_trace, offset=TRACE_OFFSET, interval="closed")
    closed_metrics = binary_metrics(closed, model_labels, model_score)
    aucs = [item["model_lag0_auc"] for item in by_offset if item["model_lag0_auc"] is not None]
    return {
        "recording": recording_id,
        "offset_frames": by_offset,
        "model_lag0_auc_range": [min(aucs), max(aucs)] if aucs else None,
        "closed_interval_model_lag0_auc": closed_metrics["auc"],
    }


def build_recording(model, recording_id: str) -> dict[str, object]:
    path = DATA / f"{recording_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    recording = build_clamped_recording(
        model, load_atanas_recording(path), expected_core=None
    )
    result = simulate_clamped(model, recording, ClampedParameters())
    events = event_vector(raw, len(recording.behavior), offset=TRACE_OFFSET)
    intervals = event_intervals(raw, len(recording.behavior), offset=TRACE_OFFSET)
    truth_score = -_z_score(recording.traces, recording.core_names)
    model_score = -_z_score(result.intrinsic, INTRINSIC_NEURONS)
    return {
        "recording_id": recording_id,
        "raw": raw,
        "recording": recording,
        "result": result,
        "events": events,
        "intervals": intervals,
        "truth_score": truth_score,
        "model_score": model_score,
        "dt": float(recording.dt),
        "n_core": int(recording.n_core),
        "n_frames": int(len(events)),
    }


def build_payload(model, recording_ids: list[str]) -> dict[str, object]:
    records = [build_recording(model, recording_id) for recording_id in recording_ids]
    by_id = {record["recording_id"]: record for record in records}
    folds = select_lag_loo(
        {
            record["recording_id"]: {
                "events": record["events"],
                "score": record["model_score"],
            }
            for record in records
        },
        shifts=DEFAULT_SHIFTS,
    )
    fold_by_id = {fold["held_out"]: fold for fold in folds}
    for fold in folds:
        fold["selected_shift_ms"] = float(
            fold["selected_shift_frames"] * by_id[fold["held_out"]]["dt"] * 1000.0
        )

    payload_records: list[dict[str, object]] = []
    for record in records:
        recording_id = str(record["recording_id"])
        recording = record["recording"]
        result = record["result"]
        events = record["events"]
        dt = float(record["dt"])
        shift = int(fold_by_id[recording_id]["selected_shift_frames"])
        readout = binary_metrics(events, recording.behavior, record["truth_score"])
        readout_ci = block_bootstrap_ci(events, recording.behavior, record["truth_score"])
        model_payload = _model_payload(
            events, result.behavior_sim, record["model_score"], shift, dt
        )
        persistence = binary_metrics(
            events,
            persistence_labels(events),
            (persistence_labels(events) == -1).astype(float),
        )
        majority = majority_labels(events)
        controls = {}
        try:
            controls = _control_payload(
                negative_controls(model, recording, ClampedParameters()), events, shift, dt
            )
        except Exception as exc:  # pragma: no cover - defensive report path
            controls = {"error": f"{type(exc).__name__}: {exc}"}
        payload_records.append(
            {
                "recording": recording_id,
                "n_core": int(record["n_core"]),
                "n_frames": int(record["n_frames"]),
                "dt_s": dt,
                "readout": _readout_payload(readout, readout_ci),
                "model": model_payload,
                "baselines": {
                    "persistence": {
                        "auc": persistence["auc"],
                        "agreement": persistence["accuracy"],
                        "precision": persistence["precision"],
                        "recall": persistence["recall"],
                    },
                    "majority": {
                        "agreement": float(np.mean((majority == -1) == events)),
                        "predicted_reverse_rate": float(np.mean(majority == -1)),
                    },
                },
                "loo": fold_by_id[recording_id],
                "controls": controls,
                "event_level": {
                    "readout": event_level_metrics(
                        events, recording.behavior, record["intervals"], dt
                    ),
                    "model_retrospective": event_level_metrics(
                        events,
                        shift_labels(result.behavior_sim, shift),
                        record["intervals"],
                        dt,
                    ),
                    "persistence": event_level_metrics(
                        events, persistence_labels(events), record["intervals"], dt
                    ),
                },
                "lag_curve": lag_curve(events, record["model_score"]),
                "prospective_max_negative_auc": max(
                    (
                        value
                        for value in (
                            lag_auc(events, record["model_score"], lag)
                            for lag in range(-20, 0)
                        )
                        if value is not None
                    ),
                    default=None,
                ),
                "alignment_sensitivity": _sensitivity(
                    record["raw"],
                    recording_id,
                    record["n_frames"],
                    result.behavior_sim,
                    record["model_score"],
                ),
            }
        )

    readout_aucs = [
        item["readout"]["auc"] for item in payload_records if item["readout"]["auc"] is not None
    ]
    model_lag0 = [
        item["model"]["lag0_auc"]
        for item in payload_records
        if item["model"]["lag0_auc"] is not None
    ]
    retrospective = [
        item["model"]["retrospective_auc"]
        for item in payload_records
        if item["model"]["retrospective_auc"] is not None
    ]
    prospective = [
        item["model"]["prospective_auc"]
        for item in payload_records
        if item["model"]["prospective_auc"] is not None
    ]
    persistence_aucs = [
        item["baselines"]["persistence"]["auc"]
        for item in payload_records
        if item["baselines"]["persistence"]["auc"] is not None
    ]
    exploratory_prospective = [
        item["prospective_max_negative_auc"]
        for item in payload_records
        if item.get("prospective_max_negative_auc") is not None
    ]
    payload = {
        "format": "vienna-tier3-behavior-validation-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "claim_level": "external behavioural sanity check; not causal, not forecasting",
        "alignment": {
            "trace_offset_frames": TRACE_OFFSET,
            "event_interval": "[start, end) 0-based",
            "lag_convention": "positive shift = event[t] vs score[t+shift] = model lags event",
            "selection": "leave-one-recording-out; held-out recording never selects the shift",
        },
        "aggregate": {
            "n_recordings": len(payload_records),
            "mean_readout_auc": float(np.mean(readout_aucs)) if readout_aucs else None,
            "mean_model_lag0_auc": float(np.mean(model_lag0)) if model_lag0 else None,
            "mean_model_retrospective_auc": float(np.mean(retrospective)) if retrospective else None,
            "mean_model_prospective_auc": float(np.mean(prospective)) if prospective else None,
            "max_exploratory_prospective_auc": (
                float(max(exploratory_prospective)) if exploratory_prospective else None
            ),
            "mean_persistence_auc": float(np.mean(persistence_aucs)) if persistence_aucs else None,
            "mean_selected_shift_frames": float(
                np.mean([fold["selected_shift_frames"] for fold in folds])
            ),
            "mean_selected_shift_ms": float(
                np.mean(
                    [
                        fold["selected_shift_frames"] * by_id[fold["held_out"]]["dt"] * 1000.0
                        for fold in folds
                    ]
                )
            ),
        },
        "folds": folds,
        "recordings": payload_records,
        "caveats": [
            "The release does not document whether reversal intervals are 0-based or 1-based; the primary convention is 0-based half-open and is checked with an offset/interval sensitivity analysis.",
            "A positive selected shift is retrospective alignment: it compares event[t] with score[t+shift], so it uses future model output and is not a forecast.",
            "The mirrored prospective AUC is below 0.5, which argues against anticipation at that shift. An exploratory maximum over all negative shifts is also reported; it is not an out-of-sample endpoint and remains below the persistence baseline.",
            "Frame-level persistence is a strong baseline because reversal is autocorrelated; the clamped model is not a validated behavioural predictor unless it beats that baseline.",
            "The endpoint is binary reversal only; the release does not provide independent labels for all eight behaviours.",
            "Controls are simulation interventions (beta=0, zero signals), not experimental causal interventions.",
        ],
        "provenance": _git_state(),
    }
    return payload


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
    model = load_premotor()
    payload = build_payload(model, args.recordings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
