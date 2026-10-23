"""Phase 2 optogenetic response benchmark on the Randi 2023 atlas.

The raw archive is not committed. Download the OSF file `9mecf` from project
`e2syt`, extract to data/raw/randi2023/exported_data/, and verify the SHA256 in
docs/INTERVENTION_DATASET_AUDIT.md before running.

This is a predictive benchmark. It does not validate NeuroBranch causally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from neurobranch.data.randi import (
    build_trial_table,
    connectome_feature_table,
    coverage_summary,
    load_connectome,
)
from neurobranch.evaluation.randi_benchmark import (
    BenchmarkConfig,
    aggregate_fold_summary,
    evaluate_randi_benchmark,
    response_reliability,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase2b_optogenetic_atlas"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "randi2023" / "exported_data")
    parser.add_argument(
        "--connectome",
        type=Path,
        default=ROOT / "references" / "external" / "aconnectome_witvliet_2020_8_utf8.csv",
    )
    parser.add_argument("--rebuild-table", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    table_path = OUT / "randi_trial_table.csv.gz"
    if args.rebuild_table or not table_path.exists():
        table = build_trial_table(args.data_dir)
        table.to_csv(table_path, index=False, compression="gzip")
    else:
        table = pd.read_csv(table_path)
    print(f"trial table: {len(table):,} rows, {table.event_id.nunique():,} events")

    coverage = coverage_summary(table, args.data_dir)
    (OUT / "randi_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    reliability = {
        "response": response_reliability(table),
        "response_residual": response_reliability(table, response_column="response_residual"),
    }
    (OUT / "response_reliability.json").write_text(
        json.dumps(reliability, indent=2), encoding="utf-8"
    )

    connectome = None
    if args.connectome.exists():
        connectome = connectome_feature_table(load_connectome(args.connectome))
        print(f"connectome features: {len(connectome)} cell types")

    for target in ["response", "response_residual"]:
        summary, oof = evaluate_randi_benchmark(
            table, connectome=connectome, cfg=BenchmarkConfig(response_column=target)
        )
        summary.to_csv(OUT / f"benchmark_summary_{target}.csv", index=False)
        oof.to_csv(OUT / f"benchmark_oof_{target}.csv.gz", index=False, compression="gzip")
        aggregate = aggregate_fold_summary(summary)
        aggregate.to_csv(OUT / f"benchmark_aggregate_{target}.csv", index=False)
        print(f"\n{target}")
        print(aggregate[["task", "model", "mae_mean", "pearson_mean", "sign_accuracy_mean"]])


if __name__ == "__main__":
    main()
