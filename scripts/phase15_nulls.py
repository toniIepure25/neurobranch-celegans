"""Derive the Phase 15 null-control table from the frozen summary.

The null draws are computed inside the audit; this helper only exposes them as a
separate artifact for the manifest and the manuscript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurobranch.benchmark.reliability_audit import ESTIMANDS

ROOT = Path(__file__).resolve().parents[1]
KEY = {"obs": "R_obs", "fold": "R_22", "id": "R_id"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "phase15")
    args = parser.parse_args(argv)
    reports = args.reports.resolve()
    summary_path = reports / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    nulls = {estimand: summary["estimands"][KEY[estimand]]["null"] for estimand in ESTIMANDS}
    (reports / "nulls.json").write_text(
        json.dumps(nulls, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    for estimand, block in nulls.items():
        print(
            f"{estimand}: permutation median {block['permutation']['median']:.4f}, "
            f"q95 {block['permutation']['q95']:.4f}; sign-flip median "
            f"{block['signflip']['median']:.4f}, q95 {block['signflip']['q95']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
