"""Download and verify the Kramer et al. (2026) directed-turning archive.

The Dryad archive is public but served through an Anubis proof-of-work gate.
This script solves the advertised SHA-256 challenge, keeps the session cookie,
and streams selected files to ``data/raw/kramer2026/files``.  Every file is
verified against the SHA-256 digest returned by Dryad's metadata API.

Raw data are written under ``data/raw`` (ignored by Git).  The generated
manifest records the provenance needed to rebuild the audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
DATASET_DOI = "doi:10.5061/dryad.8sf7m0cz2"
API = f"https://datadryad.org/api/v2/datasets/{quote(DATASET_DOI, safe='')}"
RAW = ROOT / "data" / "raw" / "kramer2026"
FILES = RAW / "files"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _get(session: requests.Session, url: str, **kwargs: Any) -> requests.Response:
    return session.get(url, timeout=kwargs.pop("timeout", 180), **kwargs)


def _solve_challenge(session: requests.Session, url: str, html: str) -> bool:
    match = re.search(
        r'<script id="anubis_challenge" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    if not match:
        return False
    challenge = json.loads(match.group(1))
    ch = challenge["challenge"]
    difficulty = int(challenge["rules"]["difficulty"])
    prefix = "0" * difficulty
    started = time.time()
    nonce = 0
    while True:
        digest = hashlib.sha256((ch["randomData"] + str(nonce)).encode()).hexdigest()
        if digest.startswith(prefix):
            break
        nonce += 1
    pass_url = "https://datadryad.org/.within.website/x/cmd/anubis/api/pass-challenge"
    response = _get(
        session,
        pass_url,
        params={
            "id": ch["id"],
            "response": digest,
            "nonce": nonce,
            "redir": url,
            "elapsedTime": int((time.time() - started) * 1000),
        },
        allow_redirects=False,
    )
    response.raise_for_status()
    return True


def _open(session: requests.Session, url: str) -> requests.Response:
    response = _get(session, url, stream=True)
    if "html" in response.headers.get("content-type", ""):
        body = response.text
        if "anubis_challenge" in body:
            response.close()
            if not _solve_challenge(session, url, body):
                raise RuntimeError(f"Could not solve Anubis challenge for {url}")
            response = _get(session, url, stream=True)
    response.raise_for_status()
    return response


def _metadata(session: requests.Session) -> dict[str, Any]:
    meta = _open(session, API).json()
    version_url = f"https://datadryad.org{meta['_links']['stash:version']['href']}"
    version = _open(session, version_url).json()
    version_id = int(version["_links"]["self"]["href"].rsplit("/", 1)[-1])
    files: list[dict[str, Any]] = []
    page = f"https://datadryad.org/api/v2/versions/{version_id}/files"
    while page:
        payload = _open(session, page).json()
        files.extend(payload["_embedded"]["stash:files"])
        page = payload["_links"].get("next", {}).get("href")
        if page:
            page = f"https://datadryad.org{page}"
    return {
        "dataset_doi": DATASET_DOI,
        "dataset_id": meta["id"],
        "title": meta["title"],
        "version_id": version_id,
        "version_number": version["versionNumber"],
        "publication_date": version["publicationDate"],
        "license": meta["license"],
        "storage_size": meta["storageSize"],
        "files": [
            {
                "path": f["path"],
                "size": f["size"],
                "sha256": f["digest"],
                "file_id": int(f["_links"]["self"]["href"].rsplit("/", 1)[-1]),
            }
            for f in sorted(files, key=lambda x: x["path"])
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="download only paths containing one of these substrings",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    FILES.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    meta = _metadata(session)
    (RAW / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    selected = [
        e
        for e in meta["files"]
        if not args.only or any(s in e["path"] for s in args.only)
    ]
    print(
        f"dataset {meta['dataset_id']} version {meta['version_id']} "
        f"{meta['storage_size'] / 1e6:.1f} MB, {len(meta['files'])} files; "
        f"selected {len(selected)}"
    )

    for entry in selected:
        target = FILES / entry["path"]
        if target.exists() and not args.force:
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest == entry["sha256"]:
                print(f"ok      {entry['path']}")
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://datadryad.org/downloads/file_stream/{entry['file_id']}"
        response = _open(session, url)
        h = hashlib.sha256()
        tmp = target.with_suffix(target.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    h.update(chunk)
        if h.hexdigest() != entry["sha256"]:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {entry['path']}")
        tmp.replace(target)
        print(f"fetched {entry['path']} ({entry['size'] / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
