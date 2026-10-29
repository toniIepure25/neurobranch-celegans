"""Phase 9: measurement ceiling audit for the Randi optogenetic atlas.

Produces the artefacts that Phase 6-8 evaluation cards should have been
bounded by:

* ``reliability_by_target.csv`` -- per-target repeat and single-animal
  reliability with animal/event counts.
* ``variance_components.json`` -- additive moment decomposition of the
  pair-level response (descriptive, not REML).
* ``reliability_summary.json`` -- aggregate ceilings with target-level 95%
  intervals.
* ``manifest.json`` -- input hashes, git revision, parameters and versions.

The script never writes a causal claim.  A ceiling is a statement about
measurement repeatability, not about mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIALS = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"
DEFAULT_OUT = ROOT / "reports" / "phase9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_table(trials: pd.DataFrame) -> pd.DataFrame:
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
    table = table.sort_values("repeat_reliability", ascending=False).reset_index(drop=True)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.trials.exists():
        print(f"trials table not found: {args.trials}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(args.trials)
    required = {"dataset_id", "target", "neuron", "event_id", "response_mean"}
    missing = required - set(trials.columns)
    if missing:
        print(f"trials table missing columns: {sorted(missing)}", file=sys.stderr)
        return 2

    table = build_table(trials)
    summary = summarise_reliability(table)
    components = decompose_variance(trials)

    table.to_csv(args.out / "reliability_by_target.csv", index=False)
    _write_json(args.out / "reliability_summary.json", summary.as_dict())
    _write_json(args.out / "variance_components.json", components.as_dict())
    _write_json(
        args.out / "manifest.json",
        {
            "phase": "phase9",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_revision": _git_revision(),
            "input": {
                "path": str(args.trials.relative_to(ROOT)) if args.trials.is_relative_to(ROOT) else str(args.trials),
                "sha256": _sha256(args.trials),
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

    payload = summary.as_dict()
    print(f"targets with repeat ceiling: {int(np.isfinite(table['repeat_reliability']).sum())}")
    print(f"targets with animal ceiling: {int(np.isfinite(table['single_animal_reliability']).sum())}")
    print(
        "repeat reliability: mean={:.3f} [{:.3f}, {:.3f}] median={:.3f}".format(
            payload["mean_repeat_reliability"],
            payload["repeat_ci_low"],
            payload["repeat_ci_high"],
            payload["median_repeat_reliability"],
        )
    )
    print(
        "single-animal reliability: mean={:.3f} [{:.3f}, {:.3f}] median={:.3f} p90={:.3f}".format(
            payload["mean_animal_reliability"],
            payload["animal_ci_low"],
            payload["animal_ci_high"],
            payload["median_animal_reliability"],
            payload["p90_animal_reliability"],
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
