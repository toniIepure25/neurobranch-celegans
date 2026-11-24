"""Resolve the prior-art identifiers used by the interventional-consistency ledger.

Usage:
    python scripts/verify_prior_art_dois.py

Reads ``research/interventional_consistency_dois.txt``, one entry per line:

    DOI | expected title fragment

The fragment is mandatory.  A DOI can resolve to the wrong work (a mistyped or
recycled identifier), so a 200 response is not verification: the resolved title
must contain the fragment.  Statuses are ``verified``, ``title_mismatch`` or an
HTTP/transport failure.  Only ``verified`` rows may be cited in a document.

Writes ``research/interventional_consistency_dois_verified.csv`` and the raw
Crossref payloads under ``research/interventional_consistency_metadata/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOIS = ROOT / "research" / "interventional_consistency_dois.txt"
DEFAULT_OUT = ROOT / "research" / "interventional_consistency_dois_verified.csv"
DEFAULT_RAW = ROOT / "research" / "interventional_consistency_metadata"
USER_AGENT = "NeuroBranch-ic-audit/1.0 (mailto:research@example.invalid)"
FIELDS = ["doi", "status", "expected_fragment", "title", "year", "container", "type"]


def fetch(doi: str) -> dict:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed host
        return json.loads(response.read().decode("utf-8"))["message"]


def parse_entries(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(f"{path.name}:{number}: missing '|' between DOI and expected title fragment")
        doi, fragment = (part.strip() for part in line.split("|", 1))
        if not doi or not fragment:
            raise ValueError(f"{path.name}:{number}: empty DOI or fragment")
        entries.append((doi, fragment))
    return entries


def _row(doi: str, fragment: str, status: str, **extra: str) -> dict[str, str]:
    row = dict.fromkeys(FIELDS, "")
    row.update({"doi": doi, "expected_fragment": fragment, "status": status})
    row.update(extra)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dois", type=Path, default=DEFAULT_DOIS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args(argv)

    if not args.dois.exists():
        print(f"missing DOI list: {args.dois}", file=sys.stderr)
        return 2

    entries = parse_entries(args.dois)
    args.raw.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for doi, fragment in entries:
        try:
            message = fetch(doi)
        except urllib.error.HTTPError as exc:
            rows.append(_row(doi, fragment, f"http_{exc.code}"))
            print(f"UNRESOLVED {doi}: HTTP {exc.code}", file=sys.stderr)
            continue
        except (urllib.error.URLError, TimeoutError) as exc:
            rows.append(_row(doi, fragment, f"error_{type(exc).__name__}"))
            print(f"UNRESOLVED {doi}: {type(exc).__name__}", file=sys.stderr)
            continue

        title = (message.get("title") or [""])[0]
        status = "verified" if fragment.lower() in title.lower() else "title_mismatch"
        issued = message.get("issued", {}).get("date-parts", [[None]])[0]
        rows.append(
            _row(
                doi,
                fragment,
                status,
                title=title,
                year=str(issued[0]) if issued else "",
                container=(message.get("container-title") or [""])[0],
                type=message.get("type", ""),
            )
        )
        slug = doi.replace("/", "_").replace(":", "_")
        (args.raw / f"{slug}.json").write_text(json.dumps(message, indent=2)[:200000], encoding="utf-8")
        print(f"{status:14} {doi}: {title[:64]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    verified = sum(1 for row in rows if row["status"] == "verified")
    print(f"{verified}/{len(rows)} verified; wrote {args.out}", file=sys.stderr)
    return 0 if verified == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
