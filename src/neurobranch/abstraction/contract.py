"""Machine-readable information contracts for neural abstraction methods.

The failure this module prevents is comparing two methods as if they differed only
in "representation quality" when one of them was handed the measurement operator,
the future, or the true latent state.  The contract makes that difference explicit
and makes the comparison refusable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHANNELS: tuple[str, ...] = (
    "current_observations",
    "observation_history",
    "future_observations",
    "behavior_labels",
    "neuron_ids",
    "missingness_mask",
    "measurement_operator",
    "true_latent_state",
    "intervention_type",
    "intervention_target",
    "intervention_magnitude",
    "test_animal_adaptation",
)

PRIVILEGED_CHANNELS: tuple[str, ...] = (
    "future_observations",
    "measurement_operator",
    "true_latent_state",
)

SCHEMA_VERSION = "ic-contract-v1"
PACKAGED_SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "information_contract.schema.json"


class ContractError(ValueError):
    """Raised when a contract is malformed or a comparison is not admissible."""


@dataclass(frozen=True)
class InformationContract:
    """Which information a method receives, declared before it is run."""

    method: str
    channels: Mapping[str, bool]
    history_length: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise ContractError("method must be a non-empty string")
        unknown = sorted(set(self.channels) - set(CHANNELS))
        missing = sorted(set(CHANNELS) - set(self.channels))
        if unknown:
            raise ContractError(f"unknown channels: {unknown}")
        if missing:
            raise ContractError(f"missing channels: {missing}")
        for name, value in self.channels.items():
            if not isinstance(value, bool):
                raise ContractError(f"channel {name!r} must be boolean, got {type(value).__name__}")
        if self.history_length is not None:
            if not isinstance(self.history_length, int) or self.history_length < 1:
                raise ContractError("history_length must be a positive integer or None")
            if not self.channels["observation_history"]:
                raise ContractError("history_length requires observation_history=true")
        if self.channels["observation_history"] and self.history_length is None:
            raise ContractError("observation_history=true requires an explicit history_length")
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"unsupported schema_version: {self.schema_version!r}")

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "channels": {name: bool(self.channels[name]) for name in CHANNELS},
            "history_length": self.history_length,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InformationContract:
        unknown = sorted(set(payload) - {"schema_version", "method", "channels", "history_length", "notes"})
        if unknown:
            raise ContractError(f"unknown contract fields: {unknown}")
        try:
            method = payload["method"]
            channels = dict(payload["channels"])
        except KeyError as exc:
            raise ContractError(f"missing required field: {exc.args[0]}") from exc
        return cls(
            method=method,
            channels=channels,
            history_length=payload.get("history_length"),
            notes=tuple(payload.get("notes", ())),
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
        )

    # -- semantics ---------------------------------------------------------
    def privileged(self) -> bool:
        return any(self.channels[name] for name in PRIVILEGED_CHANNELS)

    def label(self) -> str:
        """Coarse description used in reports; never a quality judgement."""
        if self.channels["true_latent_state"]:
            return "oracle"
        if self.privileged():
            return "privileged"
        if self.channels["behavior_labels"] and self.channels["current_observations"]:
            return "behaviour_supervised"
        if self.channels["observation_history"]:
            return "observation_history"
        if self.channels["current_observations"]:
            return "observation_only"
        return "not_defined"

    def diff(self, other: InformationContract) -> dict[str, tuple[bool, bool]]:
        """Channels whose access differs, as ``{name: (self, other)}``."""
        return {name: (self.channels[name], other.channels[name]) for name in CHANNELS if self.channels[name] != other.channels[name]}

    def is_comparable(self, other: InformationContract) -> bool:
        """Comparable as representations?  Only the history window may differ.

        ``history_length`` and the ``observation_history`` flag it implies are the
        single pre-specified exception: a method with a four-frame window and one with
        a one-frame window are compared *as* history lengths, and the report must state
        the window rather than pretend the comparison is information-matched.  Every
        other channel difference makes the comparison about information access, not
        about the representation.
        """
        exempt = {"observation_history"}
        return not {name: pair for name, pair in self.diff(other).items() if name not in exempt}

    def ranking_note(self) -> str:
        """One line for a report table, stating what the method was allowed to see."""
        seen = [name for name in CHANNELS if self.channels[name]]
        history = f"L={self.history_length}" if self.history_length else "no history"
        return f"{self.label()} [{history}]: {', '.join(seen) if seen else 'nothing'}"

    def comparison_note(self, other: InformationContract) -> str:
        if self.is_comparable(other):
            return f"{self.method} vs {other.method}: comparable information sets"
        parts = [f"{name} {a}->{b}" for name, (a, b) in self.diff(other).items()]
        return (
            f"{self.method} vs {other.method}: NOT comparable as representations "
            f"(information differs: {', '.join(parts)})"
        )


def require_comparable(contracts: Mapping[str, InformationContract]) -> None:
    """Raise unless every contract is comparable to every other."""
    items = list(contracts.items())
    for index, (name_a, contract_a) in enumerate(items):
        for name_b, contract_b in items[index + 1 :]:
            if not contract_a.is_comparable(contract_b):
                raise ContractError(
                    f"{name_a!r} and {name_b!r} cannot be compared as representations: "
                    f"{contract_a.comparison_note(contract_b)}"
                )


def load_schema(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or PACKAGED_SCHEMA).read_text(encoding="utf-8"))


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Validate a serialised contract against the packaged JSON Schema."""
    import jsonschema

    try:
        jsonschema.validate(instance=dict(payload), schema=load_schema())
    except jsonschema.ValidationError as exc:  # pragma: no cover - message formatting
        raise ContractError(f"schema validation failed: {exc.message}") from exc


def write_contracts(path: Path, contracts: list[InformationContract]) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contracts": [contract.to_dict() for contract in contracts],
    }
    for contract in contracts:
        validate_payload(contract.to_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_contracts(path: Path) -> list[InformationContract]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [InformationContract.from_dict(item) for item in payload["contracts"]]


def _contract(method: str, *, history: int | None = None, behavior: bool = False, **overrides: bool) -> InformationContract:
    channels = dict.fromkeys(CHANNELS, False)
    channels["current_observations"] = True
    if history is not None:
        channels["observation_history"] = True
    channels["behavior_labels"] = behavior
    for name, value in overrides.items():
        if name not in CHANNELS:
            raise ContractError(f"unknown channel override: {name}")
        channels[name] = value
    return InformationContract(method=method, channels=channels, history_length=history)


def stage1_contracts() -> dict[str, InformationContract]:
    """Contracts for the Stage-One reference methods, declared before any run.

    ``oracle`` and ``pca_known_H`` are privileged and excluded from rankings.
    """
    return {
        "oracle": _contract("oracle", true_latent_state=True, behavior=True),
        "pca_observation": _contract("pca_observation"),
        "pca_observation_history": _contract("pca_observation_history", history=4),
        "pca_known_H": _contract("pca_known_H", measurement_operator=True),
        "behaviour_supervised": _contract("behaviour_supervised", behavior=True),
    }
