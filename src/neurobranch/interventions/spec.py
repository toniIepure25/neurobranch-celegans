"""Virtual perturbation specifications.

An intervention is expressed in model units: standardized neural activity
(SD relative to the training split of the same worm).  This makes amplitudes
comparable across neurons but does NOT make them comparable to optogenetic
power, histamine concentration, or any physical perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from neurobranch.data.kato import BEHAVIOR_LABEL_NEURONS, MACRO_ORDER
from neurobranch.preprocessing import WormData

PerturbationMode = Literal["additive", "clamp", "zero", "scale"]


@dataclass(frozen=True)
class InterventionSpec:
    name: str
    targets: tuple[str, ...]
    mode: PerturbationMode = "additive"
    amplitude: float = 1.0
    onset: int = 0
    duration: int = 15
    horizon: int = 15
    n_initial: int = 40
    seeds: tuple[int, ...] = (0, 1, 2)
    initial_behavior: str = "REV"
    target_behavior: str = "TURN"
    notes: str = ""

    def validate(self) -> None:
        if not self.targets:
            raise ValueError("Intervention needs at least one target neuron")
        if self.mode not in {"additive", "clamp", "zero", "scale"}:
            raise ValueError(f"Unknown perturbation mode {self.mode!r}")
        if self.onset < 0 or self.duration < 0 or self.horizon < 1:
            raise ValueError("onset/duration must be non-negative and horizon positive")
        if self.initial_behavior not in MACRO_ORDER or self.target_behavior not in MACRO_ORDER:
            raise ValueError("Unknown behaviour in intervention spec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "targets": list(self.targets),
            "mode": self.mode,
            "amplitude": self.amplitude,
            "onset": self.onset,
            "duration": self.duration,
            "horizon": self.horizon,
            "n_initial": self.n_initial,
            "seeds": list(self.seeds),
            "initial_behavior": self.initial_behavior,
            "target_behavior": self.target_behavior,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InterventionSpec:
        payload = dict(payload)
        payload["targets"] = tuple(payload.get("targets", ()))
        payload["seeds"] = tuple(payload.get("seeds", (0, 1, 2)))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def target_indices(data: WormData, targets: tuple[str, ...]) -> list[int]:
    idx = []
    for name in targets:
        if name in BEHAVIOR_LABEL_NEURONS:
            raise ValueError(
                f"{name} is a behaviour-label neuron and is excluded from the predictor set; "
                "the observationally trained model cannot represent an intervention on it."
            )
        idx.append(data.neuron_index(name))
    return idx


def make_perturbation(
    data: WormData,
    spec: InterventionSpec,
) -> tuple[Any, list[int]]:
    """Return a state-edit callback and target indices."""
    spec.validate()
    inds = target_indices(data, spec.targets)

    def perturb(x: np.ndarray, step: int) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64).copy()
        if spec.onset <= step < spec.onset + spec.duration:
            if spec.mode == "additive":
                y[inds] += spec.amplitude
            elif spec.mode == "clamp":
                y[inds] = spec.amplitude
            elif spec.mode == "zero":
                y[inds] = 0.0
            elif spec.mode == "scale":
                y[inds] *= spec.amplitude
        return y

    return perturb, inds
