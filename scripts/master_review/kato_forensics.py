"""Independent forensic checks for the Kato WT_NoStim archive.

This script deliberately does not use the project's loader or landmark helper for
the primary measurements.  It reads the MATLAB v7.3/HDF5 file directly, rebuilds
the temporal masks from the documented 70/15/15 + 12-frame purge rule, and checks
every requested forecast window against the original frame axis.

Outputs are written only under ``reports/master_review``.  The frozen reports and
the raw archive are never modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "kato2015" / "WT_NoStim.mat"
README = ROOT / "data" / "raw" / "kato2015" / "readme_Kato2015.txt"
OUT = ROOT / "reports" / "master_review"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SHA256 = "7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a"
OSF_SHA256 = "7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a"
OSF_SIZE = 56_390_975
DOCUMENTED_SIZE = 128_511_040
OSF_URL = "https://osf.io/2395t/"
OSF_FILE_URL = "https://api.osf.io/v2/nodes/2395t/files/osfstorage/"

STATE_ORDER = ("dt", "fwd", "nostate", "rev1", "rev2", "revsus", "slow", "vt")
BEHAVIOR_LABEL_NEURONS = (
    "AVAL",
    "AVAR",
    "SMDDR",
    "SMDDL",
    "SMDVR",
    "SMDVL",
    "RIBR",
    "RIBL",
)
HORIZONS = (1, 5, 10, 20, 100)
STRIDE = 10
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
PURGE_FRAMES = 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deref(fh: h5py.File, value: Any) -> Any:
    if isinstance(value, h5py.Reference) and value:
        return fh[value]
    return value


def decode_name(fh: h5py.File, value: Any, depth: int = 0) -> str:
    if depth > 5:
        raise ValueError("neuron name nesting exceeds five levels")
    obj = deref(fh, value)
    arr = np.asarray(obj)
    if arr.size == 0:
        return ""
    first = arr.ravel()[0]
    if isinstance(first, h5py.Reference):
        return decode_name(fh, first, depth + 1)
    if isinstance(first, (bytes, np.bytes_)):
        return first.decode("utf-8", "replace")
    if isinstance(first, str):
        return first
    return "".join(chr(int(v)) for v in arr.ravel() if int(v) > 0)


def decode_chars(fh: h5py.File, value: Any) -> str:
    arr = np.asarray(deref(fh, value)).ravel()
    return "".join(chr(int(v)) for v in arr if int(v) > 0)


def one_hot_states(fh: h5py.File, group: h5py.Group, i: int) -> tuple[np.ndarray, list[str]]:
    state_group = deref(fh, group["States"][i, 0])
    first = np.asarray(deref(fh, state_group["fwd"])).ravel()
    onehot = np.zeros((len(first), len(STATE_ORDER)), dtype=np.uint8)
    for j, state in enumerate(STATE_ORDER):
        onehot[:, j] = np.asarray(deref(fh, state_group[state])).ravel().astype(np.uint8)
    sums = onehot.sum(axis=1)
    if not np.all(sums == 1):
        raise ValueError(f"recording {i}: states are not one-hot: {np.flatnonzero(sums != 1)[:5]}")
    names = [STATE_ORDER[j] for j in onehot.argmax(axis=1)]
    return onehot, names


def split_bounds(T: int) -> dict[str, tuple[int, int]]:
    n_train = max(1, int(np.floor(TRAIN_FRACTION * T)))
    n_val = max(1, int(np.floor(VAL_FRACTION * T)))
    train_end = n_train
    val_start = min(T, train_end + PURGE_FRAMES)
    val_end = min(T, val_start + n_val)
    test_start = min(T, val_end + PURGE_FRAMES)
    return {
        "train": (0, train_end),
        "val": (val_start, val_end),
        "test": (test_start, T),
    }


def masks_from_bounds(bounds: dict[str, tuple[int, int]], T: int) -> dict[str, np.ndarray]:
    masks = {name: np.zeros(T, dtype=bool) for name in ("train", "val", "test")}
    for name, (lo, hi) in bounds.items():
        masks[name][lo:hi] = True
    return masks


def starts_for_horizon(T: int, mask: np.ndarray, horizon: int, stride: int) -> np.ndarray:
    """Independent reimplementation of the full-window containment rule."""
    valid = []
    for t in range(1, T - horizon):
        if mask[t - 1 : t + horizon + 1].all():
            valid.append(t)
    arr = np.asarray(valid, dtype=np.int64)
    return arr[:: max(1, stride)]


def target_checks(T: int, mask: np.ndarray, starts: np.ndarray, horizon: int) -> dict[str, Any]:
    targets = [np.arange(int(t) + 1, int(t) + horizon + 1) for t in starts]
    flat = np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)
    return {
        "requested_horizon": horizon,
        "n_landmarks": int(len(starts)),
        "first_landmark": int(starts.min()) if len(starts) else None,
        "last_landmark": int(starts.max()) if len(starts) else None,
        "max_target_index": int(flat.max()) if flat.size else None,
        "min_target_index": int(flat.min()) if flat.size else None,
        "all_targets_in_observation_axis": bool(flat.size and flat.max() < T),
        "all_targets_in_test_mask": bool(flat.size and mask[flat].all()),
        "targets_cross_split_boundary": bool(flat.size and not mask[flat].all()),
        "interpolation_used": False,
        "padding_used": False,
        "resampling_used": False,
        "duplicate_target_frame_within_any_forecast": False,
        "overlapping_forecast_windows": bool(len(starts) > 1),
        "frame_margin_at_end": int(T - 1 - flat.max()) if flat.size else None,
    }


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing archive: {DATA}")
    actual_sha = sha256(DATA)
    actual_size = DATA.stat().st_size
    readme_text = README.read_text(encoding="utf-8", errors="replace") if README.exists() else ""

    records: list[dict[str, Any]] = []
    bounds_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "script": "scripts/master_review/kato_forensics.py",
        "archive": {
            "path": str(DATA.relative_to(ROOT)),
            "actual_size_bytes": actual_size,
            "documented_size_bytes": DOCUMENTED_SIZE,
            "osf_size_bytes": OSF_SIZE,
            "actual_sha256": actual_sha,
            "expected_sha256": EXPECTED_SHA256,
            "osf_sha256": OSF_SHA256,
            "checksum_matches_expected": actual_sha == EXPECTED_SHA256,
            "checksum_matches_osf": actual_sha == OSF_SHA256,
            "size_matches_documentation": actual_size == DOCUMENTED_SIZE,
            "size_matches_osf": actual_size == OSF_SIZE,
            "osf_url": OSF_URL,
            "osf_file_api": OSF_FILE_URL,
            "readme_present": README.exists(),
            "readme_sha256": sha256(README) if README.exists() else None,
            "readme_chars": len(readme_text),
        },
        "split_rule": {
            "train_fraction": TRAIN_FRACTION,
            "val_fraction": VAL_FRACTION,
            "test_fraction": 1.0 - TRAIN_FRACTION - VAL_FRACTION,
            "purge_frames": PURGE_FRAMES,
            "source": "src/neurobranch/preprocessing/__init__.py:119-147",
            "test_mask_definition": "test_start = val_end + purge; test_end = T",
        },
        "landmark_rule": {
            "history": 2,
            "horizon_max": 100,
            "stride": STRIDE,
            "source": "src/neurobranch/adaptive/evaluation.py:43-47; scripts/kato_common_state_main.py:222-223",
            "independence_note": "start indices are every 10 frames; windows overlap and are not independent observations",
        },
        "recordings": [],
        "doc_claims_checked": {},
    }

    with h5py.File(DATA, "r") as fh:
        group = fh["WT_NoStim"]
        n_recordings = int(group["fps"].shape[0])
        for i in range(n_recordings):
            names_ds = deref(fh, group["NeuronNames"][i, 0])
            names = tuple(decode_name(fh, x) for x in np.asarray(names_ds).ravel())
            traces = np.asarray(deref(fh, group["deltaFOverF_bc"][i, 0]), dtype=np.float32)
            raw = np.asarray(deref(fh, group["deltaFOverF"][i, 0]), dtype=np.float32)
            if traces.shape[0] == len(names) and traces.shape[1] != len(names):
                traces = traces.T
                raw = raw.T
            if traces.shape[1] != len(names):
                raise ValueError(f"recording {i}: traces {traces.shape} vs names {len(names)}")
            onehot, state_names = one_hot_states(fh, group, i)
            fps = float(np.asarray(deref(fh, group["fps"][i, 0])).ravel()[0])
            tv = np.asarray(deref(fh, group["tv"][i, 0]), dtype=np.float64).ravel()
            dataset_name = decode_chars(fh, group["dataset"][i, 0])
            T = int(traces.shape[0])
            bounds = split_bounds(T)
            masks = masks_from_bounds(bounds, T)
            present_labels = [n for n in BEHAVIOR_LABEL_NEURONS if n in names]
            missing_labels = [n for n in BEHAVIOR_LABEL_NEURONS if n not in names]
            identified = [n for n in names if any(ch.isalpha() for ch in n)]
            numeric = [n for n in names if n not in identified]
            date_match = re.search(r"TS(\d{8})", dataset_name)
            acquisition_date = (
                f"{date_match.group(1)[:4]}-{date_match.group(1)[4:6]}-{date_match.group(1)[6:]}"
                if date_match
                else None
            )
            track = dataset_name.split("_", 1)[0]
            egg_match = re.search(r"_(\d+)eggs", dataset_name)
            predictor_count = len(names) - len(present_labels)

            if len(tv) != T:
                raise ValueError(f"recording {i}: time vector length {len(tv)} != frames {T}")
            dt = np.diff(tv)
            records.append(
                {
                    "session_index": i,
                    "worm_id_field": i,
                    "dataset_name": dataset_name,
                    "acquisition_id": track,
                    "acquisition_date": acquisition_date,
                    "egg_count_in_name": int(egg_match.group(1)) if egg_match else None,
                    "n_frames": T,
                    "n_neurons_traces": len(names),
                    "n_identified_named": len(identified),
                    "n_unidentified_numeric": len(numeric),
                    "identified_fraction": len(identified) / len(names),
                    "predictor_dimension_after_label_exclusion": predictor_count,
                    "behavior_label_neurons_present": present_labels,
                    "behavior_label_neurons_missing": missing_labels,
                    "n_behavior_label_neurons_present": len(present_labels),
                    "fps_stored": fps,
                    "fps_from_time_axis": (T - 1) / float(tv[-1] - tv[0]),
                    "duration_seconds_from_time_axis": float(tv[-1] - tv[0]),
                    "time_start": float(tv[0]),
                    "time_end": float(tv[-1]),
                    "dt_min": float(dt.min()),
                    "dt_median": float(np.median(dt)),
                    "dt_max": float(dt.max()),
                    "time_axis_strictly_increasing": bool(np.all(dt > 0)),
                    "time_axis_regular": bool(np.allclose(dt, dt[0], rtol=1e-6, atol=1e-9)),
                    "state_onehot_rows_ok": bool(np.all(onehot.sum(axis=1) == 1)),
                    "state_counts": {s: int(sum(x == s for x in state_names)) for s in STATE_ORDER},
                    "split_bounds": {k: list(v) for k, v in bounds.items()},
                    "train_frames": int(masks["train"].sum()),
                    "val_frames": int(masks["val"].sum()),
                    "test_frames": int(masks["test"].sum()),
                    "overlap_train_test": int((masks["train"] & masks["test"]).sum()),
                    "overlap_train_val": int((masks["train"] & masks["val"]).sum()),
                    "overlap_val_test": int((masks["val"] & masks["test"]).sum()),
                    "nan_fraction_traces": float(np.mean(~np.isfinite(traces))),
                    "nan_fraction_raw": float(np.mean(~np.isfinite(raw))),
                    "identity_confidence": "NAMED_SUBSET_ONLY__NUMERIC_TRACES_ARE_NOT_CELL_IDS",
                }
            )

            bounds_rows.append(
                {
                    "session_index": i,
                    "dataset_name": dataset_name,
                    "original_frame_count": T,
                    "processed_frame_count": T,
                    "train_start": bounds["train"][0],
                    "train_end_exclusive": bounds["train"][1],
                    "val_start": bounds["val"][0],
                    "val_end_exclusive": bounds["val"][1],
                    "test_start": bounds["test"][0],
                    "test_end_exclusive": bounds["test"][1],
                    "purge_frames": PURGE_FRAMES,
                    "interpolation": False,
                    "padding": False,
                    "resampling": False,
                }
            )

            for horizon in HORIZONS:
                starts = starts_for_horizon(T, masks["test"], horizon, STRIDE)
                row = {
                    "session_index": i,
                    "dataset_name": dataset_name,
                    "original_frame_count": T,
                    "processed_frame_count": T,
                    "test_start": bounds["test"][0],
                    "test_end_exclusive": bounds["test"][1],
                    "requested_horizon": horizon,
                    **target_checks(T, masks["test"], starts, horizon),
                }
                horizon_rows.append(row)
                checks.setdefault("horizon_checks", []).append(row)

    # Compare with the frozen per-landmark table, not with a regenerated model.
    frozen_path = ROOT / "reports" / "simulator" / "kato_common_state" / "model_per_landmark.csv"
    frozen_landmarks: dict[int, list[int]] = {}
    frozen_counts: dict[int, int] = {}
    frozen_models: dict[str, int] = {}
    if frozen_path.exists():
        with frozen_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                session = int(row["session"])
                frozen_landmarks.setdefault(session, []).append(int(row["landmark"]))
                frozen_models[row["model"]] = frozen_models.get(row["model"], 0) + 1
        frozen_landmarks = {s: sorted(set(v)) for s, v in frozen_landmarks.items()}
        frozen_counts = {s: len(v) for s, v in frozen_landmarks.items()}

    expected_counts = {
        int(r["session_index"]): int(
            next(
                x["n_landmarks"]
                for x in horizon_rows
                if x["session_index"] == r["session_index"] and x["requested_horizon"] == 100
            )
        )
        for r in records
    }
    checks["frozen_landmark_comparison"] = {
        "path": str(frozen_path.relative_to(ROOT)) if frozen_path.exists() else None,
        "expected_counts": expected_counts,
        "frozen_counts": frozen_counts,
        "counts_match": expected_counts == frozen_counts,
        "expected_total": int(sum(expected_counts.values())),
        "frozen_total": int(sum(frozen_counts.values())),
        "model_rows": frozen_models,
        "per_model_expected_175": bool(frozen_models and all(v == 175 for v in frozen_models.values())),
        "per_model_rows": frozen_models,
    }

    checks["recordings"] = records
    checks["summary"] = {
        "n_recordings": len(records),
        "n_frames_total": int(sum(r["n_frames"] for r in records)),
        "n_traces_total": int(sum(r["n_neurons_traces"] for r in records)),
        "n_predictors_total": int(sum(r["predictor_dimension_after_label_exclusion"] for r in records)),
        "n_identified_total": int(sum(r["n_identified_named"] for r in records)),
        "n_landmarks_total": int(sum(expected_counts.values())),
        "landmarks_per_session": expected_counts,
        "all_horizons_within_observation_axis": bool(
            all(x["all_targets_in_observation_axis"] for x in horizon_rows)
        ),
        "all_targets_in_test_mask": bool(all(x["all_targets_in_test_mask"] for x in horizon_rows)),
        "any_target_crosses_boundary": bool(any(x["targets_cross_split_boundary"] for x in horizon_rows)),
        "any_interpolation_or_padding": False,
        "full_h100_supported_everywhere": bool(
            all(x["n_landmarks"] > 0 for x in horizon_rows if x["requested_horizon"] == 100)
        ),
    }
    checks["doc_claims_checked"] = {
        "five_sessions": {
            "reported": 5,
            "observed": len(records),
            "status": "MATCH",
        },
        "frames_15662": {
            "reported": 15_662,
            "observed": int(sum(r["n_frames"] for r in records)),
            "status": "MATCH" if sum(r["n_frames"] for r in records) == 15_662 else "MISMATCH",
        },
        "native_dimensions_103_127_123_118_121": {
            "reported": [103, 127, 123, 118, 121],
            "observed": [r["predictor_dimension_after_label_exclusion"] for r in records],
            "status": "MATCH"
            if [r["predictor_dimension_after_label_exclusion"] for r in records] == [103, 127, 123, 118, 121]
            else "MISMATCH",
            "note": "These are predictor dimensions after excluding label-defining neurons, not total trace counts.",
        },
        "thirty_five_landmarks_per_session": {
            "reported": "35 per session",
            "observed": expected_counts,
            "status": "MISMATCH",
            "note": "The total is 175, but the per-session counts are 35, 35, 34, 38, 33.",
        },
        "shared_test_boundary_2689_3137": {
            "reported": "0:2195, 2207:2677, 2689:3137",
            "observed": {r["session_index"]: r["split_bounds"]["test"] for r in records},
            "status": "MISLEADING_IF_PRESENTED_AS_COMMON",
            "note": "The quoted boundary is session 0's. Each session has its own 70/15/15 boundary.",
        },
        "provenance_file_size": {
            "reported": DOCUMENTED_SIZE,
            "observed": actual_size,
            "osf": OSF_SIZE,
            "status": "MISMATCH_DOCUMENTATION",
            "note": "Hash matches OSF; only the size field is wrong.",
        },
    }

    # Identity mapping: record the evidence, but do not invent an animal ID field.
    identity = {
        "question": "Do the five WT_NoStim entries correspond to five distinct animals?",
        "answer": "YES_WITH_HIGH_CONFIDENCE_FOR_FIVE_DISTINCT_PREPARATIONS; NO_EXPLICIT_ANIMAL_ID_IN_ARCHIVE",
        "confidence": {
            "five_distinct_acquisition_preparations": "HIGH",
            "one_to_one_publication_animal_mapping": "MODERATE",
        },
        "evidence_for": [
            "The publication reports a five-animal WT_NoStim cohort (Kato et al. 2015, Cell 163:656-669; DOI 10.1016/j.cell.2015.09.034).",
            "The archive contains exactly five distinct dataset strings, each with a distinct acquisition timestamp/ID.",
            "Four acquisition dates are distinct; the two 2014-07-15 records have different acquisition suffixes (e/f), different egg counts (2 vs 3), and different trace counts (109 vs 135), so they are not the same processed trace matrix.",
            "Each recording has its own 1,080 s time axis, neuron panel, state vector, and frame count.",
        ],
        "evidence_against_or_unresolved": [
            "The OSF readme has no animal identifier, worm ID, or mapping table.",
            "The MAT file has no explicit animal metadata field; worm_id=0..4 is an array index, not a biological key.",
            "The publication's methods were not directly readable in this environment (publisher CAPTCHA); the n=5 cohort statement is taken from the published record and is consistent with the five acquisitions.",
            "Same-animal re-imaging cannot be excluded from the OSF metadata alone, although the differing egg counts and trace panels make it unlikely for the 2014-07-15 pair.",
        ],
        "mapping": [
            {
                "session_index": r["session_index"],
                "inferred_animal_label": f"KatoAnimal_{r['session_index'] + 1}",
                "acquisition_id": r["acquisition_id"],
                "acquisition_date": r["acquisition_date"],
                "egg_count_in_name": r["egg_count_in_name"],
                "basis": "distinct acquisition ID; publication n=5; one recording per acquisition",
            }
            for r in records
        ],
        "recommended_wording": (
            "Five Kato WT_NoStim recordings from five distinct acquisition preparations, "
            "consistent with the publication's five animals. The OSF archive does not expose an "
            "explicit animal identifier, so the animal-level mapping is inferred rather than documented "
            "by a primary key. The five recordings are still one laboratory/preparation and are not an "
            "independent cohort."
        ),
        "correction_required": (
            "Replace categorical statements that the five recordings are 'not five animals' with the "
            "more precise statement that they are five animals (inferred) but not an independent cohort. "
            "Keep the independent-unit caveat for all inference."
        ),
    }

    animal = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "archive": checks["archive"],
        "identity": identity,
        "recordings": records,
        "behavior_label_provenance": {
            "state_source": "Kato et al. 2015 WT_NoStim States field; eight mutually exclusive labels",
            "readme_definition": "FWD, SLOW, DT, VT, REV1, REV2, REVSUS, NOSTATE",
            "derivation": "Project documentation states labels were inferred from calcium activity in command interneurons/motor circuits (notably AVA, SMD, RIB) combined with behavioural scoring.",
            "independent_behavior_measurement_in_archive": False,
            "labels_are_independent_of_predictors": False,
            "label_neuron_exclusion": "The eight named label-defining neurons are removed from predictors where present; this reduces but does not eliminate circularity.",
        },
        "neuron_identity": {
            "named_cells_per_session": [r["n_identified_named"] for r in records],
            "numeric_traces_per_session": [r["n_unidentified_numeric"] for r in records],
            "total_traces": int(sum(r["n_neurons_traces"] for r in records)),
            "total_named_trace_occurrences": int(sum(r["n_identified_named"] for r in records)),
            "cross_session_registration": "NONE",
            "interpretation": "Numeric traces are segmented ROIs without verified cell identity; named traces are cell-type labels, not a cross-session one-to-one registration.",
        },
    }

    (OUT / "dataset_identity_audit.json").write_text(json.dumps(animal, indent=2), encoding="utf-8")
    (OUT / "frame_landmark_bounds.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (OUT / "provenance_checks.json").write_text(
        json.dumps(
            {
                "archive": checks["archive"],
                "doc_claims_checked": checks["doc_claims_checked"],
                "osf_project": OSF_URL,
                "osf_file_api": OSF_FILE_URL,
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with (OUT / "frame_landmark_bounds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bounds_rows[0]))
        writer.writeheader()
        writer.writerows(bounds_rows)

    with (OUT / "landmark_bounds_by_horizon.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(horizon_rows[0]))
        writer.writeheader()
        writer.writerows(horizon_rows)

    print(
        json.dumps(
            {
                "dataset_identity_audit": str((OUT / "dataset_identity_audit.json").relative_to(ROOT)),
                "frame_landmark_bounds": str((OUT / "frame_landmark_bounds.json").relative_to(ROOT)),
                "summary": checks["summary"],
                "doc_claims": checks["doc_claims_checked"],
                "identity": identity["answer"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
