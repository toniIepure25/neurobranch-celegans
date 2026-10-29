"""Keep the documented root schemas in sync with the packaged copies.

The package copy is canonical: it travels with a wheel.  The root copy remains
for readers and for scripts that predate packaging.  ``--check`` fails CI on
drift rather than silently regenerating it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "neurobranch" / "schemas"
DOCUMENTED = ROOT / "schemas"


def _pairs() -> list[tuple[Path, Path]]:
    return [
        (PACKAGE / path.name, DOCUMENTED / path.name)
        for path in sorted(PACKAGE.glob("*.json"))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on drift")
    args = parser.parse_args(argv)
    failures: list[str] = []
    for source, target in _pairs():
        if not target.exists():
            failures.append(f"missing documented copy: {target.relative_to(ROOT)}")
            continue
        same = source.read_bytes() == target.read_bytes()
        if not same:
            failures.append(f"drift: {target.relative_to(ROOT)}")
            if not args.check:
                shutil.copyfile(source, target)
    if args.check and failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    if not args.check:
        print(f"synced {len(_pairs())} schemas")
    else:
        print(f"schemas in sync ({len(_pairs())} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
