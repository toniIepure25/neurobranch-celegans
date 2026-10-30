"""Phase 14A: audit the one-event variance assumption in the Phase 11 reference.

The Phase 11 estimator fills the sample variance of a one-event cell with zero.
Phase 14 replaces that value with a robust global event variance and asks whether
the reported `R(22)=0.3136` and ceiling `0.5600` survive.  This is a descriptive
audit, not a model adoption and not a new ceiling claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.benchmark.variance import (
    _components_from_arrays,
    cell_summaries,
    profile_components_from_cells,
    summarise_components,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIALS = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"
DEFAULT_OUT = ROOT / "reports" / "phase14"
REFERENCE_R = 0.3136077301426884
REFERENCE_CEILING = 0.5600069018705827
TREATMENTS = ("v0_historical", "v1_primary", "v1h_homogeneous", "v2_min_events_2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_variance(cells: pd.DataFrame, treatment: str, omega2: float) -> pd.DataFrame:
    work = cells.copy()
    if treatment == "v1_primary":
        work.loc[work["n"] <= 1, "var"] = float(omega2)
    elif treatment == "v1h_homogeneous":
        work["var"] = float(omega2)
    elif treatment not in {"v0_historical", "v2_min_events_2"}:
        raise ValueError(f"unknown treatment {treatment!r}")
    return work


def _min_events(treatment: str) -> int:
    return 2 if treatment == "v2_min_events_2" else 1


def _summary(cells: pd.DataFrame, treatment: str, omega2: float) -> tuple[dict[str, float], pd.DataFrame]:
    work = _apply_variance(cells, treatment, omega2)
    components = profile_components_from_cells(
        work,
        min_events=_min_events(treatment),
        min_animals=6,
        min_responder_coverage=0.80,
    )
    summary = summarise_components(components, fold_size=22)
    return summary, components


def _target_arrays(
    cells: pd.DataFrame, treatment: str, omega2: float
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], list[str]]:
    work = _apply_variance(cells, treatment, omega2)
    work = work[work["n"] >= _min_events(treatment)]
    animal_ids = sorted(work["_animal"].unique())
    position = {animal: index for index, animal in enumerate(animal_ids)}
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for target, group in work.groupby("_target", sort=False):
        means = group.pivot_table(index="_animal", columns="_neuron", values="mean", aggfunc="mean")
        sampling = group.assign(se2=group["var"] / np.maximum(group["n"], 1.0)).pivot_table(
            index="_animal", columns="_neuron", values="se2", aggfunc="mean"
        )
        means = means.dropna(axis=1, how="all")
        means = means.dropna(axis=0, how="all")
        if means.shape[1] < 3 or means.shape[0] < 6:
            continue
        sampling = sampling.reindex(index=means.index, columns=means.columns)
        ids = np.asarray([position[animal] for animal in means.index], dtype=int)
        arrays[str(target)] = (
            means.to_numpy(dtype=float),
            sampling.to_numpy(dtype=float),
            ids,
        )
    return arrays, animal_ids


def _bootstrap(
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_animals: int,
    *,
    n_boot: int,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for replicate in range(n_boot):
        sampled = rng.integers(0, n_animals, size=n_animals)
        components: list[dict[str, float]] = []
        for values, noise, ids in arrays.values():
            # A sampled animal may appear multiple times; searchsorted maps each
            # draw to the target-specific row and preserves duplicates.
            positions = np.searchsorted(ids, sampled)
            valid = positions < ids.size
            valid_index = np.flatnonzero(valid)
            valid[valid_index] = ids[positions[valid_index]] == sampled[valid_index]
            if not np.any(valid):
                continue
            index = positions[valid]
            fitted = _components_from_arrays(
                values[index],
                noise[index],
                min_animals=6,
                min_responder_coverage=0.80,
            )
            if fitted is None:
                continue
            components.append(
                {
                    "target": "bootstrap",
                    "n_animals": fitted.n_animals,
                    "n_responders": fitted.n_responders,
                    "signal": fitted.signal,
                    "noise": fitted.noise,
                    "interaction": fitted.interaction,
                    "pooled_reliability": fitted.pooled_reliability,
                    "single_animal_reliability": fitted.single_animal_reliability,
                }
            )
        summary = summarise_components(pd.DataFrame(components), fold_size=22)
        rows.append(
            {
                "replicate": float(replicate),
                "n_targets": float(len(components)),
                "pooled_reliability_at_fold": summary["pooled_reliability_at_fold"],
                "pooled_ceiling_at_fold": summary["pooled_ceiling_at_fold"],
            }
        )
    return rows


def _quantile(values: list[float], q: float) -> float:
    data = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.quantile(data, q)) if data.size else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args(argv)
    if not args.trials.is_file():
        print(f"trials table not found: {args.trials}", file=sys.stderr)
        return 2
    if args.n_boot < 100:
        print("--n-boot must be >= 100", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trials = pd.read_csv(args.trials)
    cells = cell_summaries(trials)

    variance_cells = cells[(cells["n"] >= 2) & (cells["var"] > 0.0)]["var"].to_numpy(dtype=float)
    if variance_cells.size == 0:
        print("no cells with n >= 2 and positive variance", file=sys.stderr)
        return 2
    omega2 = float(np.median(variance_cells))

    summaries: dict[str, dict[str, float]] = {}
    per_target: list[pd.DataFrame] = []
    for treatment in TREATMENTS:
        summary, components = _summary(cells, treatment, omega2)
        summaries[treatment] = summary
        if not components.empty:
            frame = components.copy()
            frame.insert(0, "treatment", treatment)
            per_target.append(frame)

    v0 = summaries["v0_historical"]["pooled_reliability_at_fold"]
    v1 = summaries["v1_primary"]["pooled_reliability_at_fold"]
    a1 = bool(abs(v0 - REFERENCE_R) <= 1e-6 and abs(
        summaries["v0_historical"]["pooled_ceiling_at_fold"] - REFERENCE_CEILING
    ) <= 1e-6)
    if not a1:
        print(
            "A1 failed: V0 does not reproduce the Phase 11 reference "
            f"({v0!r} vs {REFERENCE_R!r})",
            file=sys.stderr,
        )
        return 3
    a2_delta = abs(v1 - v0)
    a2 = bool(a2_delta <= 0.05)

    # Common-target summaries make the treatment comparison like-for-like.
    by_target = pd.concat(per_target, ignore_index=True) if per_target else pd.DataFrame()
    common: dict[str, Any] = {}
    if {"v0_historical", "v1_primary"}.issubset(set(by_target["treatment"])):
        v0_targets = set(by_target.loc[by_target["treatment"] == "v0_historical", "target"])
        v1_targets = set(by_target.loc[by_target["treatment"] == "v1_primary", "target"])
        shared = sorted(v0_targets & v1_targets)
        for treatment in ("v0_historical", "v1_primary"):
            subset = by_target[(by_target["treatment"] == treatment) & by_target["target"].isin(shared)]
            common[treatment] = summarise_components(subset, fold_size=22)
        common["n_shared_targets"] = len(shared)

    bootstrap_rows: list[dict[str, float]] = []
    bootstrap_summary: dict[str, Any] = {}
    for treatment in ("v0_historical", "v1_primary"):
        arrays, animal_ids = _target_arrays(cells, treatment, omega2)
        rows = _bootstrap(arrays, len(animal_ids), n_boot=args.n_boot, seed=args.seed)
        for row in rows:
            row["treatment"] = float(TREATMENTS.index(treatment))
        bootstrap_rows.extend(rows)
        bootstrap_summary[treatment] = {
            "n_boot": len(rows),
            "n_targets_median": _quantile([row["n_targets"] for row in rows], 0.5),
            "pooled_reliability_low": _quantile(
                [row["pooled_reliability_at_fold"] for row in rows], 0.025
            ),
            "pooled_reliability_high": _quantile(
                [row["pooled_reliability_at_fold"] for row in rows], 0.975
            ),
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": 14,
        "analysis": "reference_audit",
        "input": {
            "path": str(args.trials.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(args.trials),
            "redistributed": False,
        },
        "omega2_median": omega2,
        "omega2_definition": "median sample variance over cells with n>=2 and s2>0",
        "reference": {
            "pooled_reliability_at_fold": REFERENCE_R,
            "pooled_ceiling_at_fold": REFERENCE_CEILING,
            "fold_size": 22,
            "coverage": 0.80,
        },
        "treatments": summaries,
        "common_targets": common,
        "decisions": {
            "A1_reproduced": a1,
            "A2_delta": a2_delta,
            "A2_threshold": 0.05,
            "A2_reference_robust": a2,
        },
        "bootstrap": bootstrap_summary,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    (args.out / "reference_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    if not by_target.empty:
        by_target.to_csv(args.out / "reference_audit_by_target.csv", index=False)
    if bootstrap_rows:
        pd.DataFrame(bootstrap_rows).to_csv(args.out / "reference_audit_bootstrap.csv", index=False)
    print(json.dumps(payload["decisions"], indent=2))
    print(json.dumps(payload["treatments"], indent=2))
    print(f"wrote {args.out / 'reference_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
