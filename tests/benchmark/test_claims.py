"""Tests for the claims ledger."""

from __future__ import annotations

import json
from pathlib import Path

from neurobranch.benchmark.claims import build_claims, write_claims

SUMMARY = {
    "mean_animal_reliability": 0.2235,
    "animal_ci_low": 0.1963,
    "animal_ci_high": 0.2506,
    "mean_repeat_reliability": 0.2426,
    "repeat_ci_low": 0.2062,
    "repeat_ci_high": 0.2790,
}


def test_phase9_bound_claims_are_supported() -> None:
    claims = {claim.id: claim for claim in build_claims(SUMMARY)}
    assert claims["phase9-single-animal-ceiling-below-half"].status == "supported"
    assert claims["phase9-repeat-ceiling-below-half"].status == "supported"


def test_phase8_pair_claim_is_underpowered() -> None:
    claims = {claim.id: claim for claim in build_claims(SUMMARY)}
    claim = claims["phase8-pair-advantage"]
    assert claim.status == "underpowered"
    assert claim.mde is not None and claim.margin is not None
    assert claim.mde > claim.margin


def test_trajectory_and_superseded_claims_are_not_supported() -> None:
    claims = {claim.id: claim for claim in build_claims(SUMMARY)}
    assert claims["phase8-trajectory-advantage"].status == "not_supported"
    assert claims["phase7-cro-share-of-ceiling"].status == "not_supported"


def test_ledger_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "claims.json"
    write_claims(path, build_claims(SUMMARY))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["claims"]) == 7
    assert all("evidence" in claim for claim in payload["claims"])
