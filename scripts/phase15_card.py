"""Write the Phase 15 benchmark card from the audited summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "phase15")
    args = parser.parse_args(argv)
    reports = args.reports.resolve()
    summary = json.loads((reports / "summary.json").read_text(encoding="utf-8"))
    estimands = summary["estimands"]
    decision = summary["decision"]
    card: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "BoundedBench",
        "version": "2.1",
        "description": (
            "Measurement-bounded evaluation of C. elegans neural-response predictions. "
            "Reliability is reported as three separate estimands; R_22 is a counterfactual "
            "extrapolation, not an identified bound."
        ),
        "ceiling": {
            "status": "superseded_not_a_bound",
            "historical_phase11_value": 0.5600069018705827,
            "note": (
                "Phase 11 called sqrt(R(22)) the coverage-aware ceiling. Phase 15 shows that "
                "R(22) is identified for only 39 of 126 targets; 0.560 is therefore an "
                "extrapolated point estimate, not a bound. Use R_obs or R_id with their "
                "jackknife intervals and the audit label."
            ),
        },
        "estimands": {
            key: {
                "estimate": block["estimate"],
                "ceiling": block["ceiling"],
                "ci_low": block["jackknife"]["ci_low"],
                "ci_high": block["jackknife"]["ci_high"],
                "n_targets": int(block["n_targets"]),
                "n_extrapolated": int(block["n_extrapolated"]),
                "estimand": {
                    "R_obs": "observed animal counts",
                    "R_22": "counterfactual at 22 animals",
                    "R_id": "identified subset with n >= 22",
                }[key],
            }
            for key, block in estimands.items()
        },
        "decision": decision,
        "evaluation": {
            "fold_size": 22,
            "n_folds": 5,
            "primary_metric": "pair_spearman",
            "split_unit": "animal",
            "coverage": summary["parameters"]["min_responder_coverage"],
        },
        "limitations": [
            "R_22 is extrapolated for 87 of 126 targets and the half-split extrapolation gate failed.",
            "Delete-one-animal intervals are reported only with the calibration status in the audit.",
            "Null controls were run; read their medians and q95 values, not only the point estimate.",
            "Target rankings did not replicate and are not published.",
            "No causal, transport, digital-twin, foundation-model, or animal-budget claim is supported.",
            "The derived Randi trial table is not redistributed until its licence is confirmed.",
        ],
        "provenance": {
            "summary": "reports/phase15/summary.json",
            "manifest": "reports/phase15/manifest.json",
        },
    }
    (reports / "benchmark_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {reports / 'benchmark_card.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
