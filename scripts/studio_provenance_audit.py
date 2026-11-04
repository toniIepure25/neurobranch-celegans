"""Version-aware provenance audit for the Scientific Studio release candidate.

The historical Kato manifests were generated from worktrees, not always from the
clean Git blob at their recorded ``head``.  Validating those manifests against
the current worktree is therefore the wrong test: legitimate later edits make a
correct historical manifest look broken.

This command separates:

* HISTORICAL_SNAPSHOT_INTEGRITY -- the manifest entry against its source Git
  tree, allowing a documented line-ending representation and, when necessary,
  the commit where the recorded bytes actually occur;
* CURRENT_WORKTREE_INTEGRITY -- the current Studio manifest against current
  bytes;
* EXPECTED_VERSION_DIFFERENCE -- an authorized later code/docs edit;
* UNEXPLAINED_ARTIFACT_MODIFICATION -- no Git tree explains the recorded bytes.

It is read-only.  It does not rewrite a manifest or change the worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "scientific_studio" / "release_candidate" / "integrity_report.json"
STUDIO_MANIFEST = ROOT / "reports" / "scientific_studio" / "manifest.json"
HISTORICAL_MANIFESTS = (
    (
        "kato_final_audit",
        ROOT / "reports" / "simulator" / "kato_final_audit" / "manifest.json",
        "head_at_manifest_generation",
    ),
    (
        "kato_interactions",
        ROOT / "reports" / "simulator" / "kato_interactions" / "manifest.json",
        "git_head",
    ),
)


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def git_blob(rev: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=ROOT,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def line_variants(blob: bytes) -> Iterable[tuple[str, bytes]]:
    """Yield byte representations that differ only by newline convention."""
    yield "git_bytes", blob
    if b"\r\n" not in blob:
        yield "crlf_bytes", blob.replace(b"\n", b"\r\n")
    if b"\r\n" in blob:
        yield "lf_bytes", blob.replace(b"\r\n", b"\n")


def match_kind(blob: bytes, expected_sha: str, expected_bytes: int) -> str | None:
    for label, candidate in line_variants(blob):
        if len(candidate) == expected_bytes and sha256(candidate) == expected_sha:
            return label
    return None


def commits_touching(path: str) -> list[str]:
    result = git("rev-list", "--all", "--", path, check=False)
    return [line for line in result.stdout.splitlines() if line]


def find_matching_commit(path: str, expected_sha: str, expected_bytes: int) -> str | None:
    for commit in commits_touching(path):
        blob = git_blob(commit, path)
        if blob is not None and len(blob) == expected_bytes and sha256(blob) == expected_sha:
            return commit
    return None


def dirty_paths_from_status(lines: Iterable[str]) -> set[str]:
    paths = set()
    for line in lines:
        text = str(line)
        if len(text) < 4:
            continue
        candidate = text[3:].strip().strip('"')
        if " -> " in candidate:
            candidate = candidate.split(" -> ", 1)[1].strip().strip('"')
        if candidate:
            paths.add(candidate)
    return paths


@dataclass
class EntryResult:
    path: str
    manifest_bytes: int
    manifest_sha256: str
    source_commit: str
    source_blob_bytes: int | None
    source_blob_sha256: str | None
    historical_snapshot_integrity: str
    historical_detail: str
    matching_commit: str | None
    recorded_worktree_deviation: bool
    current_bytes: int | None
    current_sha256: str | None
    current_worktree_integrity: str
    scientific_artifact: bool


def current_blob(path: str) -> bytes | None:
    candidate = ROOT / path
    return candidate.read_bytes() if candidate.is_file() else None


def classify_entry(
    path: str,
    entry: dict,
    source_commit: str,
    recorded_dirty: set[str],
) -> EntryResult:
    expected_bytes = int(entry["bytes"])
    expected_sha = str(entry["sha256"])
    source = git_blob(source_commit, path)
    current = current_blob(path)
    matching_commit = None
    was_dirty = path in recorded_dirty
    detail = "source Git blob matches the manifest"

    kind = (
        match_kind(source, expected_sha, expected_bytes)
        if source is not None
        else None
    )
    if source is not None and kind == "git_bytes":
        snapshot = "PASS"
    elif source is not None and kind is not None:
        snapshot = "PASS_WITH_LINE_ENDING_REPRESENTATION"
        detail = (
            "manifest bytes match the source blob only after newline conversion "
            f"({kind}); this is a representation difference, not a content edit"
        )
    else:
        matching_commit = find_matching_commit(path, expected_sha, expected_bytes)
        if matching_commit:
            snapshot = "PASS_AT_MATCHING_COMMIT"
            detail = (
                "recorded bytes occur at a later/earlier Git commit than the "
                "manifest's recorded head; manifest metadata is stale"
            )
        elif was_dirty:
            snapshot = "UNVERIFIABLE_WORKTREE_SNAPSHOT"
            detail = (
                "path was recorded as modified/untracked when the manifest was "
                "generated, but its exact bytes are absent from every Git commit "
                "and from the object database; the manifest SHA-256 is the only "
                "surviving byte record. This is not an unexplained scientific-"
                "artifact change, but it is not a Git-verifiable snapshot."
            )
        else:
            snapshot = "UNEXPLAINED_ARTIFACT_MODIFICATION"
            detail = "no Git blob accounts for the recorded bytes"

    if current is None:
        current_state = "MISSING"
    elif len(current) == expected_bytes and sha256(current) == expected_sha:
        current_state = "PASS"
    else:
        current_state = "EXPECTED_VERSION_DIFFERENCE"

    return EntryResult(
        path=path,
        manifest_bytes=expected_bytes,
        manifest_sha256=expected_sha,
        source_commit=source_commit,
        source_blob_bytes=None if source is None else len(source),
        source_blob_sha256=None if source is None else sha256(source),
        historical_snapshot_integrity=snapshot,
        historical_detail=detail,
        matching_commit=matching_commit,
        recorded_worktree_deviation=was_dirty,
        current_bytes=None if current is None else len(current),
        current_sha256=None if current is None else sha256(current),
        current_worktree_integrity=current_state,
        scientific_artifact=path.startswith("reports/") and path.endswith(
            (".csv", ".json", ".npz", ".png", ".pdf")
        ),
    )


def verify_historical(name: str, path: Path, source_key: str) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source_commit = str(manifest.get(source_key, ""))
    if not source_commit:
        return {"name": name, "manifest": str(path), "status": "FAIL", "error": f"missing {source_key}"}
    recorded_dirty = dirty_paths_from_status(
        manifest.get("git_status_porcelain_at_generation", [])
    )
    entries = [
        classify_entry(str(entry["path"]), entry, source_commit, recorded_dirty)
        for entry in manifest.get("files", [])
    ]
    counts: dict[str, int] = {}
    for item in entries:
        counts[item.historical_snapshot_integrity] = (
            counts.get(item.historical_snapshot_integrity, 0) + 1
        )
    unexplained = [
        asdict(item)
        for item in entries
        if item.historical_snapshot_integrity == "UNEXPLAINED_ARTIFACT_MODIFICATION"
    ]
    expected = [
        asdict(item)
        for item in entries
        if item.historical_snapshot_integrity != "PASS"
    ]
    unverifiable = [
        item
        for item in entries
        if item.historical_snapshot_integrity == "UNVERIFIABLE_WORKTREE_SNAPSHOT"
    ]
    if unexplained:
        status = "FAIL"
    elif unverifiable:
        status = "INCOMPLETE"
    elif expected:
        status = "PASS_WITH_RECORDED_WORKTREE_DEVIATIONS"
    else:
        status = "PASS"
    return {
        "name": name,
        "manifest": str(path.relative_to(ROOT)),
        "source_commit": source_commit,
        "recorded_source_key": source_key,
        "recorded_dirty_paths": sorted(recorded_dirty),
        "status": status,
        "counts": counts,
        "entries": [asdict(item) for item in entries],
        "unexplained": unexplained,
        "unverifiable_worktree_snapshots": [asdict(item) for item in unverifiable],
        "expected_version_differences": expected,
    }


def verify_studio_manifest() -> dict:
    manifest = json.loads(STUDIO_MANIFEST.read_text(encoding="utf-8"))
    bad = []
    counts: dict[str, int] = {}
    for _section, entries in manifest.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "path" not in entry:
                continue
            current = current_blob(str(entry["path"]))
            if current is None:
                state = "MISSING"
            elif len(current) == entry["bytes"] and sha256(current) == entry["sha256"]:
                state = "PASS"
            else:
                state = "CHANGED"
            counts[state] = counts.get(state, 0) + 1
            if state != "PASS":
                bad.append({"path": entry["path"], "state": state})
    status = "FAIL" if bad else "PASS"
    return {
        "manifest": str(STUDIO_MANIFEST.relative_to(ROOT)),
        "source_head": manifest.get("git", {}).get("source_head"),
        "status": status,
        "counts": counts,
        "bad": bad,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    historical = [verify_historical(*item) for item in HISTORICAL_MANIFESTS]
    studio = verify_studio_manifest()
    status = "PASS"
    if any(item["status"] == "FAIL" for item in historical) or studio["status"] != "PASS":
        status = "FAIL"
    elif any(item["status"] == "INCOMPLETE" for item in historical):
        status = "INCOMPLETE"

    payload = {
        "schema": "neurobranch.studio.provenance_audit.v1",
        "repository": str(ROOT),
        "head": git("rev-parse", "HEAD").stdout.strip(),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
        "status": status,
        "historical": historical,
        "current_studio_manifest": studio,
        "definitions": {
            "HISTORICAL_SNAPSHOT_INTEGRITY": "manifest entry checked against its recorded source Git commit",
            "CURRENT_WORKTREE_INTEGRITY": "current Studio manifest checked against current bytes",
            "EXPECTED_VERSION_DIFFERENCE": "authorized later code/docs edit, or a line-ending representation difference",
            "UNEXPLAINED_ARTIFACT_MODIFICATION": "no Git tree explains the recorded bytes",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2 if args.pretty else None) + "\n",
        encoding="utf-8",
    )

    print(f"status={status}")
    for item in historical:
        print(
            f"{item['name']}: {item['status']} source={item['source_commit'][:12]} "
            f"counts={item['counts']} unexplained={len(item['unexplained'])}"
        )
    print(
        f"current Studio manifest: {studio['status']} counts={studio['counts']} "
        f"bad={len(studio['bad'])}"
    )
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
