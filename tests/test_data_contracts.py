"""Contracts for incoming cohorts and the dataset registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobranch.data.schemas import CohortManifest, DatasetSpec, SchemaError

SHA = "a" * 64


def _manifest(**overrides: object) -> CohortManifest:
    payload: dict[str, object] = {
        "cohort_id": "sixth-worm",
        "source": "lab release",
        "license": "CC-BY-4.0",
        "sha256": SHA,
        "unit": "animal",
        "n_units": 1,
        "timebase_s": 0.5,
        "behaviors": ("forward", "reverse"),
        "intervention": False,
        "files": ("traces.npy", "metadata.json"),
        "allowed_uses": ("external_only",),
        "owner": "Denis",
        "notes": "",
    }
    payload.update(overrides)
    return CohortManifest(**payload)  # type: ignore[arg-type]


def test_dataset_spec_keeps_backward_compatible_defaults() -> None:
    spec = DatasetSpec(
        key="kato2015",
        citation="Kato et al.",
        license=None,
        raw_glob="data/raw/kato2015/WT_NoStim.mat",
        loader="neurobranch.data.kato.load_kato",
        timebase_s=0.34,
        modalities=("calcium",),
        allowed_uses=("train", "eval"),
    )
    assert spec.required_columns == ()
    assert spec.claim_level == "external_only"
    assert spec.owner is None


def test_dataset_spec_accepts_new_contract_fields() -> None:
    spec = DatasetSpec(
        key="new",
        citation="new",
        license="CC0-1.0",
        raw_glob="data/incoming/new/*.npz",
        loader="new.load",
        timebase_s=0.5,
        modalities=("calcium", "behaviour"),
        allowed_uses=("external_only",),
        required_columns=("animal_id", "frame"),
        expected_units="animal",
        claim_level="descriptive",
        owner="Denis",
    )
    assert spec.to_dict()["required_columns"] == ("animal_id", "frame")
    assert spec.claim_level == "descriptive"


def test_dataset_spec_rejects_unknown_claim_level() -> None:
    with pytest.raises(SchemaError, match="claim_level"):
        DatasetSpec(
            key="new",
            citation="new",
            license="CC0-1.0",
            raw_glob="x",
            loader="x",
            timebase_s=1.0,
            modalities=("calcium",),
            allowed_uses=("external_only",),
            claim_level="causal-proof",
        )


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "incoming_manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    loaded = CohortManifest.from_path(path)
    assert loaded == manifest


def test_manifest_rejects_string_instead_of_file_list() -> None:
    payload = _manifest().to_dict()
    payload["files"] = "traces.npy"
    with pytest.raises(SchemaError, match="files"):
        CohortManifest.from_dict(payload)


def test_manifest_rejects_bad_sha() -> None:
    with pytest.raises(SchemaError, match="sha256"):
        _manifest(sha256="not-a-hash")


def test_manifest_rejects_absolute_or_parent_paths() -> None:
    with pytest.raises(SchemaError, match="relative"):
        _manifest(files=("C:/secret.npy",))
    with pytest.raises(SchemaError, match="relative"):
        _manifest(files=("../../secret.npy",))


def test_manifest_rejects_duplicate_files() -> None:
    with pytest.raises(SchemaError, match="duplicate"):
        _manifest(files=("a.npy", "a.npy"))


def test_manifest_rejects_train_use_without_licence() -> None:
    with pytest.raises(SchemaError, match="licen"):
        _manifest(license=None, allowed_uses=("train",))


def test_manifest_rejects_empty_cohort_and_bad_unit() -> None:
    with pytest.raises(SchemaError, match="cohort_id"):
        _manifest(cohort_id="")
    with pytest.raises(SchemaError, match="unit"):
        _manifest(unit="worm")
