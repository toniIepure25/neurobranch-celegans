"""Audit the saved Phase 8 predictions for paired-inference alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from neurobranch.benchmark.inference import audit_paired_inputs

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports" / "phase8" / "predictions.parquet"
OUT = ROOT / "reports" / "phase12r" / "inference_alignment_audit.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if not args.predictions.is_file():
        print(f"missing predictions: {args.predictions}")
        return 2
    audit = audit_paired_inputs(pd.read_parquet(args.predictions))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
