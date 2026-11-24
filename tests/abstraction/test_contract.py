"""Information contract: validation, labels and refusal of unfair comparisons."""

from __future__ import annotations

import json

import pytest

from neurobranch.abstraction.contract import (
    CHANNELS,
    ContractError,
    InformationContract,
    load_contracts,
    require_comparable,
    stage1_contracts,
    validate_payload,
    write_contracts,
)


def _channels(**overrides: bool) -> dict[str, bool]:
    channels = dict.fromkeys(CHANNELS, False)
    channels["current_observations"] = True
    channels.update(overrides)
    return channels


def test_stage1_contracts_cover_channels_and_labels() -> None:
    contracts = stage1_contracts()
    assert set(contracts) == {
        "oracle",
        "pca_observation",
        "pca_observation_history",
        "pca_known_H",
        "behaviour_supervised",
    }
    assert contracts["oracle"].label() == "oracle"
    assert contracts["pca_observation"].label() == "observation_only"
    assert contracts["pca_observation_history"].label() == "observation_history"
    assert contracts["pca_known_H"].label() == "privileged"
    assert contracts["behaviour_supervised"].label() == "behaviour_supervised"
    for contract in contracts.values():
        validate_payload(contract.to_dict())


def test_privileged_methods_are_flagged() -> None:
    contracts = stage1_contracts()
    assert contracts["oracle"].privileged()
    assert contracts["pca_known_H"].privileged()
    assert not contracts["pca_observation"].privileged()
    assert not contracts["behaviour_supervised"].privileged()


def test_unknown_or_missing_channel_is_rejected() -> None:
    with pytest.raises(ContractError, match="unknown channels"):
        InformationContract(method="x", channels={**_channels(), "telepathy": True})
    channels = _channels()
    channels.pop("neuron_ids")
    with pytest.raises(ContractError, match="missing channels"):
        InformationContract(method="x", channels=channels)


def test_non_boolean_channel_is_rejected() -> None:
    with pytest.raises(ContractError, match="must be boolean"):
        InformationContract(method="x", channels={**_channels(), "neuron_ids": 1})  # type: ignore[dict-item]


def test_history_flag_requires_an_explicit_length() -> None:
    with pytest.raises(ContractError, match="requires an explicit history_length"):
        InformationContract(method="x", channels=_channels(observation_history=True))
    with pytest.raises(ContractError, match="requires observation_history"):
        InformationContract(method="x", channels=_channels(), history_length=4)


def test_cross_contract_comparison_is_refused() -> None:
    contracts = stage1_contracts()
    with pytest.raises(ContractError, match="cannot be compared as representations"):
        require_comparable(
            {"pca": contracts["pca_observation"], "oracle": contracts["oracle"]}
        )


def test_history_length_is_the_only_exempt_difference() -> None:
    short = InformationContract(method="a", channels=_channels(), history_length=None)
    long = InformationContract(
        method="b", channels=_channels(observation_history=True), history_length=8
    )
    assert short.is_comparable(long)
    require_comparable({"short": short, "long": long})
    assert long.comparison_note(short).endswith("comparable information sets")


def test_contract_roundtrip(tmp_path) -> None:
    contracts = list(stage1_contracts().values())
    path = tmp_path / "contracts.json"
    write_contracts(path, contracts)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "ic-contract-v1"
    restored = load_contracts(path)
    assert [contract.to_dict() for contract in restored] == [contract.to_dict() for contract in contracts]


def test_schema_rejects_unknown_fields() -> None:
    payload = stage1_contracts()["pca_observation"].to_dict()
    payload["extra"] = True
    with pytest.raises(ContractError, match="schema validation failed"):
        validate_payload(payload)


def test_ranking_note_states_what_was_visible() -> None:
    note = stage1_contracts()["pca_observation_history"].ranking_note()
    assert "observation_history" in note
    assert "L=4" in note
