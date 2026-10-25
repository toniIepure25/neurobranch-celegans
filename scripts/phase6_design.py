"""Build model-dependent experiment candidates from CRO predictions."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from neurobranch.causal.design import design_experiments
from neurobranch.config import PROJECT_ROOT, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="reports/phase6/cro/predictions.parquet")
    parser.add_argument("--output-dir", default="reports/phase6/design")
    parser.add_argument("--top-targets", type=int, default=12)
    args = parser.parse_args()
    path = resolve_path(args.predictions, PROJECT_ROOT)
    predictions = pd.read_parquet(path)
    summary = design_experiments(
        predictions,
        output_dir=resolve_path(args.output_dir, PROJECT_ROOT),
        top_targets=args.top_targets,
    )
    print(json.dumps(summary, indent=2)[:5000])


if __name__ == "__main__":
    main()
