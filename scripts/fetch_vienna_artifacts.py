"""Copy the third-party research artifacts used by the Vienna workbench.

The artifacts are intentionally not committed: the upstream repositories carry
different licenses, and the NC-MCM/premotor files are large.  This script copies
them from the local reference checkouts, verifies their SHA-256 hashes, and
writes a machine-readable manifest.

Usage:
    python scripts/fetch_vienna_artifacts.py
    python scripts/fetch_vienna_artifacts.py --verify-only
    python scripts/fetch_vienna_artifacts.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references" / "external"
PROCESSED = ROOT / "data" / "processed"
MANIFEST = PROCESSED / "vienna_artifacts_manifest.json"

ARTIFACTS = [
    {
        "tier": "tier1_bundlenet",
        "name": "bundlenet_consistent_embedding_worm_{i}.npz",
        "source": REFERENCES / "nc-mcm" / "precomputed_results" / "bundlenet_consistent_embedding_worm_{i}.npz",
        "destination": PROCESSED / "bundlenet_sim" / "bundlenet_consistent_embedding_worm_{i}.npz",
        "indices": range(5),
        "license": "NC-MCM repository: GPL-3.0 (precomputed embedding)",
    },
    {
        "tier": "tier2_ncmcm",
        "name": "nc_mcm_model_worm_{i}.npz",
        "source": REFERENCES / "nc-mcm" / "precomputed_results" / "nc_mcm_model_worm_{i}.npz",
        "destination": PROCESSED / "ncmcm" / "nc_mcm_model_worm_{i}.npz",
        "indices": range(1, 6),
        "license": "NC-MCM repository: GPL-3.0",
    },
    {
        "tier": "tier3_premotor",
        "name": "avg_model_weights.jld2",
        "source": REFERENCES
        / "Celegans_premotor"
        / "data"
        / "model_weights"
        / "avg_model_weights.jld2",
        "destination": PROCESSED / "premotor" / "avg_model_weights.jld2",
        "indices": [None],
        "license": "Celegans_premotor repository: MIT",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(item: dict[str, object], index: int | None) -> tuple[Path, Path]:
    if index is None:
        return Path(str(item["source"])), Path(str(item["destination"]))
    return (
        Path(str(item["source"]).format(i=index)),
        Path(str(item["destination"]).format(i=index)),
    )


def build_records(force: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    missing: list[Path] = []
    for item in ARTIFACTS:
        for index in item["indices"]:
            source, destination = resolve(item, index)
            if not source.exists():
                missing.append(source)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if force or not destination.exists() or sha256(source) != sha256(destination):
                shutil.copy2(source, destination)
            records.append(
                {
                    "tier": item["tier"],
                    "destination": destination.relative_to(ROOT).as_posix(),
                    "source": source.relative_to(ROOT).as_posix(),
                    "sha256": sha256(destination),
                    "bytes": destination.stat().st_size,
                    "license": item["license"],
                }
            )
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing reference artifacts. Clone the upstream repositories into "
            f"references/external/ first:\n{lines}"
        )
    return records


def verify(records: list[dict[str, object]]) -> None:
    failures = []
    for record in records:
        path = ROOT / str(record["destination"])
        if not path.exists() or sha256(path) != record["sha256"]:
            failures.append(str(record["destination"]))
    if failures:
        raise SystemExit("Hash verification failed for:\n" + "\n".join(f"  - {p}" for p in failures))
    print(f"verified {len(records)} artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="only check an existing manifest")
    parser.add_argument("--force", action="store_true", help="overwrite destination files")
    args = parser.parse_args()

    if args.verify_only:
        if not MANIFEST.exists():
            raise SystemExit(f"manifest not found: {MANIFEST}")
        verify(json.loads(MANIFEST.read_text(encoding="utf-8"))["artifacts"])
        return

    records = build_records(force=args.force)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_root": ".",
        "note": "Third-party artifacts; not committed. Re-run this script to restore them.",
        "artifacts": records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify(records)
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
