"""Verify the integrity and provenance of generated Phase 10 artifacts.

Usage:
    python scripts/check_reproducibility.py
    python scripts/check_reproducibility.py --reports DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "phase10")
    parser.add_argument(
        "--allow-missing-input",
        action="store_true",
        help="skip the input hash when the dataset is not present (CI checkouts)",
    )
    args = parser.parse_args(argv)

    manifest_path = args.reports / "manifest.json"
    if not manifest_path.exists():
        print(f"missing manifest: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checksum_path = args.reports / "manifest.sha256"
    if checksum_path.exists():
        expected_manifest = checksum_path.read_text(encoding="utf-8").split()[0]
        if sha256(manifest_path) != expected_manifest:
            failures.append("manifest.sha256 does not match manifest.json")
        else:
            print("ok: manifest.sha256")

    input_info = manifest.get("input") or {}
    input_path = input_info.get("path")
    if input_path:
        candidate = ROOT / Path(input_path)
        if not candidate.exists():
            if args.allow_missing_input:
                print(f"skip: input {input_path} is not present")
            else:
                failures.append(f"input missing: {input_path}")
        elif input_info.get("sha256") and sha256(candidate) != input_info["sha256"]:
            failures.append(f"input hash mismatch: {input_path}")
        else:
            print(f"ok: input {input_path}")

    artifacts = manifest.get("artifacts") or {}
    if not artifacts:
        failures.append("manifest has no artifact hashes")
    for name, expected in sorted(artifacts.items()):
        path = args.reports / name
        if not path.exists():
            failures.append(f"artifact missing: {name}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"artifact hash mismatch: {name}")
        else:
            print(f"ok: {name}")

    summary_path = args.reports / "reliability_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        version = summary.get("schema_version")
        if version != 2:
            failures.append(f"reliability summary schema_version is {version}, expected 2")
        else:
            print("ok: reliability summary schema v2")
        if summary.get("fold_size", 0) < 1:
            failures.append("fold_size must be positive")

    claims_path = args.reports / "claims.json"
    if claims_path.exists():
        claims = json.loads(claims_path.read_text(encoding="utf-8"))
        if claims.get("schema_version") != 2:
            failures.append("claims schema_version is not 2")
        else:
            print("ok: claims schema v2")

    phase9_hash = manifest.get("phase9_manifest_sha256")
    phase9_path = ROOT / "reports" / "phase9" / "manifest.json"
    if phase9_hash and phase9_path.exists() and sha256(phase9_path) != phase9_hash:
        failures.append("phase9 manifest hash mismatch")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("reproducibility checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
