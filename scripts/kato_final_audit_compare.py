"""Compare the clean rerun with the frozen committed artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "reports" / "simulator" / "kato_common_state"
RERUN = ROOT / "reports" / "simulator" / "kato_final_audit" / "clean_rerun"
OUT = ROOT / "reports" / "simulator" / "kato_final_audit"

FILES = {
    "model_per_landmark.csv": ["session", "landmark", "model"],
    "model_summary.csv": ["model"],
    "paired_contrasts.csv": ["model", "reference", "metric", "block"],
    "decomposition.csv": ["model", "reference", "metric", "block", "contrast"],
    "session_effects.csv": ["session"],
    "horizon_profile.csv": ["model", "horizon"],
    "factor_variance.csv": ["session", "k"],
    "representation_sensitivity.csv": ["model", "reference", "metric", "representation"],
    "stage_a_comparison.csv": ["model", "prior_model", "metric"],
    "surrogate_draws.csv": ["session", "landmark", "method", "draw", "seed"],
    "surrogate_paired.csv": ["model", "reference", "metric", "block"],
    "surrogate_summary.csv": ["method"],
    "synthetic_audit.csv": ["scenario", "seed"],
    "synthetic_summary.csv": ["scenario"],
}


def compare_frame(name: str, keys: list[str]) -> dict:
    a = pd.read_csv(COMMITTED / name)
    b = pd.read_csv(RERUN / name)
    for key in keys:
        if key not in a.columns or key not in b.columns:
            return {"file": name, "status": "MISSING_KEY", "missing_key": key}
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    if len(a) != len(b):
        return {"file": name, "status": "ROW_COUNT_MISMATCH", "committed_rows": len(a), "rerun_rows": len(b)}
    merged = a.merge(b, on=keys, how="outer", suffixes=("_committed", "_rerun"), indicator=True)
    if (merged["_merge"] != "both").any():
        return {"file": name, "status": "KEY_MISMATCH", "n_unmatched": int((merged["_merge"] != "both").sum())}
    numeric = [
        c
        for c in a.columns
        if c not in keys and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])
    ]
    max_abs = 0.0
    max_rel = 0.0
    worst = None
    mismatch = 0
    for c in numeric:
        av = merged[f"{c}_committed"].to_numpy(dtype=np.float64)
        bv = merged[f"{c}_rerun"].to_numpy(dtype=np.float64)
        delta = np.abs(av - bv)
        finite = np.isfinite(delta)
        if not finite.any():
            continue
        cur_abs = float(np.max(delta[finite]))
        denom = np.maximum(np.maximum(np.abs(av), np.abs(bv)), 1e-12)
        cur_rel = float(np.max(delta[finite] / denom[finite]))
        n_bad = int(np.sum(delta[finite] > 1e-8))
        mismatch += n_bad
        if cur_abs > max_abs:
            max_abs = cur_abs
            max_rel = cur_rel
            worst = c
    status = "IDENTICAL_WITHIN_1E-8" if max_abs <= 1e-8 else ("CLOSE_WITHIN_1E-6" if max_abs <= 1e-6 else "DIFFERENT")
    return {
        "file": name,
        "status": status,
        "rows": len(a),
        "numeric_columns": len(numeric),
        "max_abs_difference": max_abs,
        "max_relative_difference_at_worst": max_rel,
        "worst_column": worst,
        "values_over_1e-8": mismatch,
    }


def key_values(path: Path) -> dict:
    summary = pd.read_csv(path / "model_summary.csv").set_index("model")
    paired = pd.read_csv(path / "paired_contrasts.csv")
    row = paired[
        (paired["model"] == "B5_common_k10")
        & (paired["reference"] == "B3_population")
        & (paired["metric"] == "pca10_energy_h20")
        & (paired["block"] == 5)
    ].iloc[0]
    return {
        "B3_energy": float(summary.loc["B3_population", "energy_h20"]),
        "B5_energy": float(summary.loc["B5_common_k10", "energy_h20"]),
        "B3_advantage_B5_minus_B3": float(-row["mean_improvement"]),
        "ci_low": float(-row["ci_high"]),
        "ci_high": float(-row["ci_low"]),
        "fraction_b3_better": float(1.0 - row["fraction_better"]),
        "n_pairs": int(row["n_pairs"]),
    }


def compare_json(name: str) -> dict:
    a = json.loads((COMMITTED / name).read_text(encoding="utf-8"))
    b = json.loads((RERUN / name).read_text(encoding="utf-8"))
    if name == "gates.json":
        keys = sorted(set(a) | set(b))
        mismatched = [k for k in keys if a.get(k, {}).get("status") != b.get(k, {}).get("status")]
        return {
            "file": name,
            "status": "GATE_STATUS_MATCH" if not mismatched else "GATE_STATUS_MISMATCH",
            "gate_statuses": {k: b.get(k, {}).get("status") for k in keys},
            "mismatched_gates": mismatched,
        }
    return {"file": name, "status": "EXACT_JSON_MATCH" if a == b else "JSON_DIFFERS"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comparisons = [compare_frame(name, keys) for name, keys in FILES.items() if (RERUN / name).exists()]
    comparisons.extend(
        [compare_json(name) for name in ("gates.json", "synthetic_verification.json", "reproduction_summary.json") if (RERUN / name).exists()]
    )
    missing = [name for name in FILES if not (RERUN / name).exists()]
    pd.DataFrame(comparisons).to_csv(OUT / "comparison_numeric.csv", index=False)
    if missing:
        status = "FULL_END_TO_END_REPRODUCTION_INCOMPLETE"
    else:
        failed = [
            r
            for r in comparisons
            if r.get("status")
            not in (
                "IDENTICAL_WITHIN_1E-8",
                "CLOSE_WITHIN_1E-6",
                "GATE_STATUS_MATCH",
                "EXACT_JSON_MATCH",
            )
        ]
        close = [r for r in comparisons if r.get("status") == "CLOSE_WITHIN_1E-6"]
        status = (
            "FULL_END_TO_END_REPRODUCTION_PASSED"
            if not failed and not close
            else ("FULL_END_TO_END_REPRODUCTION_PASSED_WITH_TOLERANCE" if not failed else "FULL_END_TO_END_REPRODUCTION_FAILED")
        )
    committed_keys = key_values(COMMITTED)
    rerun_keys = key_values(RERUN) if (RERUN / "model_summary.csv").exists() and (RERUN / "paired_contrasts.csv").exists() else {}
    key_diffs = {
        k: {
            "committed": committed_keys[k],
            "rerun": rerun_keys.get(k),
            "abs_difference": abs(committed_keys[k] - rerun_keys[k]) if k in rerun_keys else None,
        }
        for k in committed_keys
    }
    payload = {
        "schema": "neurobranch.kato_final_audit.reproduction.v1",
        "status": status,
        "tolerance": {
            "deterministic_abs": 1e-8,
            "close_abs": 1e-6,
            "rationale": "same code, data, seeds and float64/CPU environment; 1e-8 is a conservative deterministic tolerance, 1e-6 is the historical G1 tolerance",
        },
        "committed_directory": str(COMMITTED.relative_to(ROOT)),
        "rerun_directory": str(RERUN.relative_to(ROOT)),
        "missing_files": missing,
        "comparisons": comparisons,
        "key_metrics": key_diffs,
        "all_numeric_within_1e-8": all(
            r.get("status") in ("IDENTICAL_WITHIN_1E-8", "GATE_STATUS_MATCH", "EXACT_JSON_MATCH")
            for r in comparisons
        )
        if comparisons and not missing
        else False,
    }
    (OUT / "reproduction_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
