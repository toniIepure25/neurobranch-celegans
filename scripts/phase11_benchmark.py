"""Phase 11: benchmark card, fold-level leaderboard and frozen split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from neurobranch.benchmark.leaderboard import build_leaderboard, leaderboard_summary
from neurobranch.benchmark.reporting import ROOT

DEFAULT_METRICS = ROOT / "reports" / "phase8" / "metrics.csv"
DEFAULT_PREDICTIONS = ROOT / "reports" / "phase8" / "predictions.parquet"
DEFAULT_OUT = ROOT / "reports" / "phase11"
FALLBACK_CEILING = 0.8327336863740054


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _split_manifest(predictions: Path, out: Path) -> dict[str, Any] | None:
    if not predictions.exists():
        return None
    frame = pd.read_parquet(predictions, columns=["dataset_id", "fold"])
    unique = frame.drop_duplicates().sort_values(["dataset_id", "fold"]).reset_index(drop=True)
    duplicates = unique.groupby("dataset_id")["fold"].nunique()
    if (duplicates > 1).any():
        raise ValueError("an animal appears in more than one fold")
    payload = unique.to_csv(index=False).encode("utf-8")
    (out / "splits.csv").write_bytes(payload)
    return {
        "path": "reports/phase11/splits.csv",
        "sha256": _sha256_bytes(payload),
        "n_animals": int(len(unique)),
        "n_folds": int(unique["fold"].nunique()),
        "folds": sorted(int(value) for value in unique["fold"].unique()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not args.metrics.exists():
        print(f"metrics not found: {args.metrics}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    summary_path = args.out / "summary.json"
    phase11_summary: dict[str, Any] = {}
    if summary_path.exists():
        phase11_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sensitivity_path = args.out / "coverage_sensitivity.csv"
    ceiling = FALLBACK_CEILING
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
        row = sensitivity[sensitivity["min_responder_coverage"] == 0.8]
        if not row.empty:
            ceiling = float(row.iloc[0]["pooled_ceiling_at_fold"])
    elif phase11_summary.get("headline", {}).get("coverage_0_8_pooled_reliability"):
        ceiling = float(phase11_summary["headline"]["coverage_0_8_pooled_reliability"])
    if not ceiling or ceiling <= 0:
        ceiling = FALLBACK_CEILING

    metrics = pd.read_csv(args.metrics)
    leaderboard = build_leaderboard(metrics, ceiling=ceiling)
    leaderboard.to_csv(args.out / "leaderboard.csv", index=False)
    board_summary = leaderboard_summary(leaderboard)
    (args.out / "leaderboard.json").write_text(
        json.dumps(board_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    splits = _split_manifest(args.predictions, args.out)
    claims_path = ROOT / "reports" / "phase10" / "claims.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8")) if claims_path.exists() else {}
    claim_statuses = {
        str(item["id"]): str(item["status"]) for item in claims.get("claims", []) if "id" in item
    }
    card = {
        "benchmark": "BoundedBench",
        "version": "2.0",
        "schema_version": 1,
        "description": (
            "Measurement-bounded evaluation of C. elegans neural-response predictions. "
            "Scores are descriptive and must be read against the reliability ceiling."
        ),
        "datasets": [
            {
                "name": "Randi et al. 2023 optogenetic atlas (derived trial table)",
                "path": "reports/phase3/randi_trial_table_phase3.csv.gz",
                "redistributed": False,
                "license_status": "unresolved; see docs/LICENSE_AUDIT.md",
            },
            {
                "name": "Kato et al. 2015 whole-brain calcium imaging",
                "path": "data/raw/kato2015/WT_NoStim.mat",
                "sha256": "7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a",
                "redistributed": False,
            },
        ],
        "evaluation": {
            "split_unit": "animal",
            "fold_size": int(claims.get("fold_size", 22)) if claims else 22,
            "n_folds": 5,
            "split_seed": 41,
            "split_manifest": splits,
            "primary_metric": "pair_spearman",
            "secondary_metrics": ["profile_spearman", "coverage", "interval_score"],
        },
        "ceiling": {
            "estimand": "pooled_fold_profile",
            "value": ceiling,
            "source": "sqrt(coverage-aware pooled reliability at n_eval=22, coverage 0.8)",
            "note": (
                "Phase 10 sparse-coverage estimate (0.833) is not used as the card default; "
                "coverage sensitivity is reported in coverage_sensitivity.csv."
            ),
        },
        "claim_statuses": claim_statuses,
        "leaderboard": board_summary,
        "limitations": [
            "Fold-level scores share animals and are not animal-generalisation estimates.",
            "Virtual perturbation outputs are model-dependent and not causally validated.",
            "The derived Randi trial table is not redistributed until its licence is confirmed.",
            "Coverage filtering changes the responder population; 0.560 is not a bias-corrected 0.833.",
            "The two-way bootstrap is biased on sparse real data and is not a calibrated interval.",
            "The extrapolation gate failed; the curve beyond observed animals is not validated.",
            "Target rankings do not replicate and are not published.",
        ],
    }
    (args.out / "benchmark_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ceiling": ceiling, "leaderboard": board_summary, "splits": splits}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
