"""Build the Phase 9 claims ledger from the reliability summary.

The ledger is generated, never hand-edited.  Each entry carries its estimate,
interval, smallest effect of interest, design MDE and evidence paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neurobranch.benchmark.claims import build_claims, write_claims

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "reports" / "phase9" / "reliability_summary.json"
DEFAULT_OUT = ROOT / "reports" / "phase9" / "claims.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="verify evidence paths exist")
    args = parser.parse_args(argv)

    if not args.summary.exists():
        print(f"summary not found: {args.summary}", file=sys.stderr)
        return 2
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    claims = build_claims(summary)
    write_claims(args.out, claims)

    width = max(len(claim.id) for claim in claims)
    for claim in claims:
        print(f"{claim.id:<{width}}  {claim.status}")
    print(f"wrote {args.out}")

    if args.check:
        missing = [
            path
            for claim in claims
            for path in claim.evidence
            if not (ROOT / path).exists()
        ]
        if missing:
            print(f"missing evidence: {sorted(set(missing))}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
