"""Build and verify the Phase 10 claims ledger.

Usage:
    python scripts/phase10_claims.py
    python scripts/phase10_claims.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neurobranch.benchmark.claims import build_claims, write_claims
from neurobranch.benchmark.reporting import ROOT

DEFAULT_SUMMARY = ROOT / "reports" / "phase10" / "reliability_summary.json"
DEFAULT_OUT = ROOT / "reports" / "phase10"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="verify evidence paths instead of rewriting")
    args = parser.parse_args(argv)

    if not args.summary.exists():
        print(f"summary not found: {args.summary}", file=sys.stderr)
        return 2
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    claims = build_claims(summary)

    missing: list[str] = []
    for claim in claims:
        for evidence in claim.evidence:
            candidate = ROOT / evidence.rstrip("/")
            if not candidate.exists():
                missing.append(f"{claim.id}: {evidence}")

    if args.check:
        if missing:
            for item in missing:
                print(f"MISSING {item}", file=sys.stderr)
            return 1
        for claim in claims:
            print(f"{claim.id:42s} {claim.status}")
        print(f"ok: {len(claims)} claims, all evidence paths exist")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    write_claims(args.out / "claims.json", claims)
    for claim in claims:
        print(f"{claim.id:42s} {claim.status}")
    if missing:
        for item in missing:
            print(f"warning: missing evidence {item}", file=sys.stderr)
    print(f"wrote {args.out / 'claims.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
