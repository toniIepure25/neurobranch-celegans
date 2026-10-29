"""Report builders shared by the CLI and the Phase 9 scripts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.benchmark.variance import (
    animal_reliability_by_target,
    decompose_variance,
    repeat_reliability_by_target,
    summarise_reliability,
)

ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(cwd: Path = ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_reliability_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Per-target repeat and across-animal reliability, merged on target."""
    repeat = repeat_reliability_by_target(trials)
    animal = animal_reliability_by_target(trials)
    table = repeat.merge(
        animal.rename(
            columns={
                "n_animals": "n_animals_animal",
                "n_responders": "n_responders_animal",
            }
        ),
        on="target",
        how="outer",
    )
    table["n_animals"] = table["n_animals"].fillna(table["n_animals_animal"])
    table["n_responders"] = table["n_responders"].fillna(table["n_responders_animal"])
    table = table.drop(columns=["n_animals_animal", "n_responders_animal"])
    columns = [
        "target",
        "n_animals",
        "n_events",
        "n_responders",
        "repeat_reliability",
        "repeat_ceiling",
        "single_animal_reliability",
        "animal_mean_reliability",
        "animal_ceiling",
    ]
    table = table[columns]
    numeric = [column for column in columns if column != "target"]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="coerce")
    table[numeric] = table[numeric].round(6)
    return table.sort_values("repeat_reliability", ascending=False).reset_index(drop=True)


def write_reliability_report(trials: pd.DataFrame, out: Path, *, input_path: Path | None = None) -> dict[str, Any]:
    """Write the four Phase 9 reliability artefacts and return the summary."""
    out.mkdir(parents=True, exist_ok=True)
    table = build_reliability_table(trials)
    summary = summarise_reliability(table)
    components = decompose_variance(trials)

    table.to_csv(out / "reliability_by_target.csv", index=False)
    write_json(out / "reliability_summary.json", summary.as_dict())
    write_json(out / "variance_components.json", components.as_dict())
    write_json(
        out / "manifest.json",
        {
            "phase": "phase9",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_revision": git_revision(),
            "input": {
                "path": (
                    str(input_path.relative_to(ROOT))
                    if input_path is not None and input_path.is_relative_to(ROOT)
                    else (str(input_path) if input_path is not None else None)
                ),
                "sha256": sha256_file(input_path) if input_path is not None and input_path.exists() else None,
                "rows": int(len(trials)),
                "animals": int(trials["dataset_id"].nunique()),
                "targets": int(trials["target"].nunique()),
            },
            "parameters": {
                "repeat_min_events": 2,
                "repeat_min_responders": 3,
                "animal_min_events": 1,
                "animal_min_animals": 6,
                "ci": "target-level normal approximation (descriptive)",
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "notes": [
                "Reliability is a measurement ceiling, not a causal effect.",
                "Target-level intervals describe the target population and ignore animal sharing.",
            ],
        },
    )
    return summary.as_dict()


__all__ = [
    "build_reliability_table",
    "clean_json",
    "git_revision",
    "sha256_file",
    "write_json",
    "write_reliability_report",
]
