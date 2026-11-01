"""Raw versus canonical-text hash behavior."""

from __future__ import annotations

import json
from pathlib import Path

from neurobranch.reproduction.hashes import (
    hash_record,
    sha256_canonical_text,
    sha256_raw,
    verify_hash,
    verify_manifest_records,
)


def test_raw_and_canonical_hashes_differ_for_crlf(tmp_path: Path):
    p = tmp_path / "report.json"
    p.write_bytes(b'{"a": 1}\r\n')
    assert sha256_raw(p) != sha256_canonical_text(p)


def test_canonical_hash_matches_lf_copy(tmp_path: Path):
    crlf = tmp_path / "crlf.txt"
    lf = tmp_path / "lf.txt"
    crlf.write_bytes(b"a\r\nb\r\n")
    lf.write_bytes(b"a\nb\n")
    assert sha256_canonical_text(crlf) == sha256_raw(lf)


def test_record_declares_mode_and_verification_rejects_mode_swap(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"a\r\nb\r\n")
    rec = hash_record(p, mode="canonical_text")
    assert rec["normalization"] == "canonical_text"
    assert verify_hash(p, rec)[0]
    assert not verify_hash(p, rec, mode="raw")[0]
    raw = hash_record(p, mode="raw")
    assert not verify_hash(p, raw, mode="canonical_text")[0]


def test_manifest_verifier_keeps_legacy_raw_compatibility(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"abc")
    manifest = {"artifacts": [{"path": "x.txt", "sha256": sha256_raw(p)}]}
    out = verify_manifest_records(manifest, root=tmp_path)
    assert out["n_ok"] == 1
    assert out["results"][0]["mode"].startswith("legacy_raw")


def test_manifest_verifier_accepts_explicit_raw_field(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"abc")
    manifest = {
        "artifacts": [
            {
                "path": "x.txt",
                "sha256_raw": sha256_raw(p),
                "algorithm": "sha256",
                "normalization": "raw",
            }
        ]
    }
    out = verify_manifest_records(manifest, root=tmp_path)
    assert out["n_ok"] == 1
    assert out["results"][0]["mode"] == "raw"


def test_canonical_manifest_roundtrip(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"a\r\nb\r\n")
    manifest = {"artifacts": [hash_record(p, mode="canonical_text")]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = verify_manifest_records(manifest, root=tmp_path)
    assert out["n_ok"] == 1
