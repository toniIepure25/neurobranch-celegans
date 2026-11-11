"""Validation for incoming cohorts, annotations and intervention trials.

These checks are intentionally conservative.  A validator may return a structured
report for data-quality problems, but structural errors (missing columns, bad
frame intervals, duplicate rows) raise ``ValueError`` so a caller cannot silently
treat malformed data as a negative result.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from neurobranch.data.schemas import CohortManifest

ANNOTATION_COLUMNS = {
    "recording_id",
    "frame_start",
    "frame_end",
    "behavior_label",
    "annotator_id",
    "confidence",
    "notes",
}
ANNOTATION_REQUIRED = ANNOTATION_COLUMNS - {"notes"}

TRIAL_COLUMNS = {
    "animal_id",
    "session_id",
    "arm",
    "target",
    "sham_type",
    "intervention_start_frame",
    "intervention_end_frame",
    "behavior_label_path",
    "neural_trace_path",
    "fps",
    "clock_offset_ms",
    "annotator_id",
    "notes",
}
TRIAL_REQUIRED = TRIAL_COLUMNS - {"notes"}
ARMS = {"target", "sham", "random-neuron", "unperturbed"}


def _read_rows(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise ValueError(f"{path}: missing columns for {label}: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no {label} rows")
    return rows


def _as_int(value: str, field: str, row_number: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {field} must be an integer, got {value!r}") from exc


def _as_float(value: str, field: str, row_number: int) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"row {row_number}: {field} must be finite")
    return out


def fleiss_kappa(counts: list[list[int]]) -> float:
    """Fleiss' kappa for a fixed number of raters per subject.

    ``counts[i][j]`` is the number of raters assigning subject ``i`` to category
    ``j``.  Returns ``nan`` when kappa is undefined (one category or zero
    variance).
    """
    if not counts:
        return float("nan")
    n = len(counts)
    k = len(counts[0])
    if k < 2 or any(len(row) != k for row in counts):
        return float("nan")
    raters = sum(counts[0])
    if raters < 2 or any(sum(row) != raters for row in counts):
        return float("nan")
    p_j = [sum(row[j] for row in counts) / (n * raters) for j in range(k)]
    p_i = [
        (sum(value * value for value in row) - raters) / (raters * (raters - 1))
        for row in counts
    ]
    p_bar = sum(p_i) / n
    p_e = sum(value * value for value in p_j)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(1.0 - p_bar) < 1e-12 else float("nan")
    return (p_bar - p_e) / (1.0 - p_e)


def _validate_annotation_rows(
    rows: Iterable[dict[str, str]],
    *,
    source: str,
    kappa_threshold: float = 0.6,
    require_kappa: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    intervals: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    annotators: set[str] = set()
    recordings: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        recording = (row.get("recording_id") or "").strip()
        annotator = (row.get("annotator_id") or "").strip()
        behavior = (row.get("behavior_label") or "").strip()
        if not recording:
            raise ValueError(f"{source}: row {row_number}: recording_id is empty")
        if not annotator:
            raise ValueError(f"{source}: row {row_number}: annotator_id is empty")
        if not behavior:
            raise ValueError(f"{source}: row {row_number}: behavior_label is empty")
        start = _as_int(row.get("frame_start", ""), "frame_start", row_number)
        end = _as_int(row.get("frame_end", ""), "frame_end", row_number)
        confidence = _as_float(row.get("confidence", ""), "confidence", row_number)
        if start < 0:
            raise ValueError(f"{source}: row {row_number}: frame_start must be non-negative")
        if end <= start:
            raise ValueError(f"{source}: row {row_number}: frame_end must exceed frame_start")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{source}: row {row_number}: confidence must be in [0, 1]")
        intervals[(recording, annotator)].append((start, end, behavior))
        annotators.add(annotator)
        recordings.add(recording)

    for (recording, annotator), spans in intervals.items():
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"{source}: overlapping intervals for {recording}/{annotator}: "
                    f"{previous[:2]} and {current[:2]}"
                )

    recording_reports: dict[str, Any] = {}
    kappas: list[float] = []
    for recording in sorted(recordings):
        by_annotator = {
            annotator: spans
            for (candidate, annotator), spans in intervals.items()
            if candidate == recording
        }
        if len(by_annotator) < 2:
            errors.append(f"{recording}: two annotators are required")
            continue
        frame_labels: dict[str, dict[int, str]] = {}
        for annotator, spans in by_annotator.items():
            labels: dict[int, str] = {}
            for start, end, behavior in spans:
                for frame in range(start, end):
                    labels[frame] = behavior
            frame_labels[annotator] = labels
        common = set.intersection(*(set(labels) for labels in frame_labels.values()))
        if not common:
            errors.append(f"{recording}: annotators have no overlapping annotated frames")
            continue
        evaluable = [frame for frame in sorted(common) if all(
            frame_labels[annotator][frame] != "unclear" for annotator in frame_labels
        )]
        unclear_frames = len(common) - len(evaluable)
        if not evaluable:
            errors.append(f"{recording}: all overlapping frames are unclear")
            continue
        categories = sorted(
            {frame_labels[annotator][frame] for frame in evaluable for annotator in frame_labels}
        )
        counts: list[list[int]] = []
        for frame in evaluable:
            row_counts = [0] * len(categories)
            for annotator in frame_labels:
                row_counts[categories.index(frame_labels[annotator][frame])] += 1
            counts.append(row_counts)
        kappa = fleiss_kappa(counts)
        p_bar = sum(
            (sum(value * value for value in row) - len(frame_labels)) /
            (len(frame_labels) * (len(frame_labels) - 1))
            for row in counts
        ) / len(counts)
        pabak = 2.0 * p_bar - 1.0
        if math.isnan(kappa):
            warnings.append(f"{recording}: kappa undefined")
        else:
            kappas.append(kappa)
        recording_reports[recording] = {
            "annotators": sorted(by_annotator),
            "n_common_frames": len(common),
            "n_evaluable_frames": len(evaluable),
            "unclear_rate": unclear_frames / len(common),
            "kappa": None if math.isnan(kappa) else kappa,
            "pabak": pabak,
            "categories": categories,
        }
        if unclear_frames / len(common) > 0.2:
            warnings.append(f"{recording}: unclear rate exceeds 20%")

    macro_kappa = sum(kappas) / len(kappas) if kappas else None
    if len(annotators) < 2:
        errors.append("two annotators are required for agreement")
    if require_kappa and (macro_kappa is None or macro_kappa < kappa_threshold):
        errors.append(
            f"macro kappa {macro_kappa!r} is below the pre-registered threshold "
            f"{kappa_threshold}"
        )
    return {
        "format": "vienna-annotation-qc-v1",
        "source": source,
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_recordings": len(recordings),
        "n_annotators": len(annotators),
        "macro_kappa": macro_kappa,
        "kappa_threshold": kappa_threshold,
        "recordings": recording_reports,
    }


def validate_annotation_csv(
    path: str | Path,
    *,
    kappa_threshold: float = 0.6,
    require_kappa: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    rows = _read_rows(path, ANNOTATION_REQUIRED, "annotation")
    return _validate_annotation_rows(
        rows,
        source=str(path),
        kappa_threshold=kappa_threshold,
        require_kappa=require_kappa,
    )


def validate_annotation_directory(
    directory: str | Path,
    *,
    kappa_threshold: float = 0.6,
    require_kappa: bool = False,
) -> dict[str, Any]:
    directory = Path(directory)
    paths = sorted(
        path
        for path in directory.rglob("*.csv")
        if path.name not in {"annotation_manifest.csv", "agreement_report.csv"}
    )
    if not paths:
        raise FileNotFoundError(f"{directory}: no annotation CSV files")
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_rows(path, ANNOTATION_REQUIRED, "annotation"))
    return _validate_annotation_rows(
        rows,
        source=str(directory),
        kappa_threshold=kappa_threshold,
        require_kappa=require_kappa,
    )


def validate_intervention_trials(
    path: str | Path,
    *,
    min_per_arm: int = 1,
    check_paths: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    if min_per_arm < 1:
        raise ValueError("min_per_arm must be positive")
    rows = _read_rows(path, TRIAL_REQUIRED, "intervention trial")
    errors: list[str] = []
    warnings: list[str] = []
    arm_counts: Counter[str] = Counter()
    arm_animals: dict[str, set[str]] = defaultdict(set)
    animals: set[str] = set()
    sessions: set[str] = set()
    seen: set[tuple[str, str, int]] = set()

    for row_number, row in enumerate(rows, start=2):
        animal = (row.get("animal_id") or "").strip()
        session = (row.get("session_id") or "").strip()
        arm = (row.get("arm") or "").strip()
        if not animal:
            raise ValueError(f"row {row_number}: animal_id is empty")
        if not session:
            raise ValueError(f"row {row_number}: session_id is empty")
        if arm not in ARMS:
            raise ValueError(f"row {row_number}: unknown arm {arm!r}")
        target = (row.get("target") or "").strip()
        sham_type = (row.get("sham_type") or "").strip()
        if arm == "target" and not target:
            raise ValueError(f"row {row_number}: target arm needs a target neuron")
        if arm == "sham" and not sham_type:
            raise ValueError(f"row {row_number}: sham arm needs a sham_type")
        start = _as_int(row.get("intervention_start_frame", ""), "intervention_start_frame", row_number)
        end = _as_int(row.get("intervention_end_frame", ""), "intervention_end_frame", row_number)
        fps = _as_float(row.get("fps", ""), "fps", row_number)
        if start < 0 or end <= start:
            raise ValueError(f"row {row_number}: invalid intervention frame interval")
        if fps <= 0:
            raise ValueError(f"row {row_number}: fps must be positive")
        clock_raw = (row.get("clock_offset_ms") or "").strip()
        if not clock_raw:
            errors.append(f"row {row_number}: clock_offset_ms is empty")
        else:
            _as_float(clock_raw, "clock_offset_ms", row_number)
        for field in ("behavior_label_path", "neural_trace_path"):
            if not (row.get(field) or "").strip():
                raise ValueError(f"row {row_number}: {field} is empty")
            if check_paths:
                candidate = Path(row[field])
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                if not candidate.exists():
                    errors.append(f"row {row_number}: {field} does not exist: {candidate}")
        key = (animal, session, start)
        if key in seen:
            errors.append(f"row {row_number}: duplicate trial {key}")
        seen.add(key)
        arm_counts[arm] += 1
        arm_animals[arm].add(animal)
        animals.add(animal)
        sessions.add(session)

    if "target" not in arm_counts:
        errors.append("no target arm")
    if "sham" not in arm_counts:
        errors.append("no sham arm")
    for arm, animal_set in sorted(arm_animals.items()):
        if len(animal_set) < min_per_arm:
            errors.append(
                f"arm {arm!r} has {len(animal_set)} animals; minimum is {min_per_arm} "
                "(trials within one animal are not independent)"
            )
    if len(animals) < 2:
        warnings.append("fewer than two independent animals; this is not a group comparison")
    return {
        "format": "vienna-intervention-trial-qc-v1",
        "source": str(path),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_trials": len(rows),
        "n_animals": len(animals),
        "n_sessions": len(sessions),
        "arm_counts": dict(sorted(arm_counts.items())),
        "animal_counts": {arm: len(values) for arm, values in sorted(arm_animals.items())},
        "min_per_arm": min_per_arm,
    }


def validate_new_worm_manifest(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Gate a sixth-worm delivery before the Tier-1 loader sees it.

    Shape/label checks stay in ``scripts/validate_new_worm.py``.  This function
    checks the provenance contract: one animal, external-only use, observational
    data, declared behaviours, and files that actually exist.
    """
    manifest_path = Path(manifest_path)
    manifest = CohortManifest.from_path(manifest_path)
    errors: list[str] = []
    root = Path(data_root) if data_root is not None else manifest_path.parent
    if manifest.unit != "animal":
        errors.append(f"unit must be 'animal', got {manifest.unit!r}")
    if manifest.n_units != 1:
        errors.append(f"a sixth-worm delivery must contain exactly one animal, got {manifest.n_units}")
    if manifest.intervention:
        errors.append("the sixth held-out worm must be observational, not an intervention cohort")
    if not manifest.behaviors:
        errors.append("no behaviour labels declared")
    if "train" in manifest.allowed_uses:
        errors.append("the sixth worm is external-only and must not be used for training")
    if "external_only" not in manifest.allowed_uses:
        errors.append("allowed_uses must include external_only")
    for relative in manifest.files:
        candidate = root / relative
        if not candidate.exists():
            errors.append(f"declared file does not exist: {candidate}")
    return {
        "format": "vienna-sixth-worm-manifest-v1",
        "source": str(manifest_path),
        "cohort_id": manifest.cohort_id,
        "passed": not errors,
        "errors": errors,
        "manifest": manifest.to_dict(),
    }
