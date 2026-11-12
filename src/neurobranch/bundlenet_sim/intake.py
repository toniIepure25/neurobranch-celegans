"""In-session validation of incoming files.

Streamlit uploads are bytes, while the validators read paths.  These helpers
write the bytes to a short-lived temporary directory, run the same validators
used by the CLI, and replace temporary paths in the report with the uploaded
file names.  Nothing is persisted after the call.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from neurobranch.data.validation import (
    validate_annotation_csv,
    validate_annotation_directory,
    validate_intervention_trials,
)

MAX_UPLOAD_BYTES = 10_000_000


def _check_csv(name: str, data: bytes) -> None:
    if not name.lower().endswith(".csv"):
        raise ValueError(f"{name}: only CSV files are accepted")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"{name}: file is too large (limit {MAX_UPLOAD_BYTES} bytes)")
    if not data:
        raise ValueError(f"{name}: file is empty")


def _safe_name(name: str, index: int) -> str:
    base = Path(name.replace("\\", "/")).name
    return f"{index:02d}_{base}"


def validate_annotation_uploads(
    files: list[tuple[str, bytes]],
    *,
    kappa_threshold: float = 0.6,
    require_kappa: bool = True,
) -> dict:
    """Validate one or more blinded annotation CSVs and compare annotators."""
    if not files:
        raise ValueError("at least one annotation file is required")
    for name, data in files:
        _check_csv(name, data)
    with TemporaryDirectory(prefix="neurobranch-annotations-") as temp:
        root = Path(temp)
        for index, (name, data) in enumerate(files):
            (root / _safe_name(name, index)).write_bytes(data)
        if len(files) == 1:
            report = validate_annotation_csv(
                next(root.glob("*.csv")),
                kappa_threshold=kappa_threshold,
                require_kappa=require_kappa,
            )
        else:
            report = validate_annotation_directory(
                root,
                kappa_threshold=kappa_threshold,
                require_kappa=require_kappa,
            )
    report["source"] = ", ".join(name for name, _ in files)
    return report


def validate_trial_upload(
    name: str,
    data: bytes,
    *,
    min_per_arm: int = 6,
) -> dict:
    """Validate a pre-unblinding intervention trial CSV."""
    _check_csv(name, data)
    with TemporaryDirectory(prefix="neurobranch-trials-") as temp:
        path = Path(temp) / _safe_name(name, 0)
        path.write_bytes(data)
        report = validate_intervention_trials(path, min_per_arm=min_per_arm)
    report["source"] = name
    return report
