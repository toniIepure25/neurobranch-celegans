"""Typed objects for the NeuroBranch neural dynamics simulator.

The simulator is deliberately more explicit than the underlying dynamics
models.  A rollout owns its initial state, seed, noise level, perturbation
protocol and support diagnostics.  This prevents a teacher-forced prediction
from being reported as a free-running simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

PerturbationProtocol = Literal["initial_state", "dynamical_input", "empirical_response"]
PerturbationMode = Literal["additive", "clamp", "zero", "scale"]

WARNING = (
    "MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED. "
    "This is a simulation under stated model assumptions, not evidence that "
    "the same perturbation controls a living animal."
)


@dataclass(frozen=True)
class SimulatorSpec:
    """Configuration of one simulator model instance."""

    model_kind: str
    history: int = 1
    noise_std: float = 0.0
    seed: int = 17
    checkpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.history < 1:
            raise ValueError("history must be at least 1")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")


@dataclass(frozen=True)
class PerturbationSpec:
    """A named-neuron perturbation in standardised model units.

    ``initial_state`` changes only the supplied initial condition.  It is a
    mathematical sensitivity probe, not stimulation.  ``dynamical_input`` adds
    an explicit input during a window of the rollout.  ``empirical_response``
    is only legal when a matched stimulation-response model has passed the
    empirical evidence gate.
    """

    target: str
    protocol: PerturbationProtocol = "dynamical_input"
    mode: PerturbationMode = "additive"
    amplitude: float = 1.0
    onset: int = 0
    duration: int = 12

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("Perturbation target is required")
        if self.protocol not in {"initial_state", "dynamical_input", "empirical_response"}:
            raise ValueError(f"Unknown perturbation protocol {self.protocol!r}")
        if self.mode not in {"additive", "clamp", "zero", "scale"}:
            raise ValueError(f"Unknown perturbation mode {self.mode!r}")
        if self.onset < 0 or self.duration < 0:
            raise ValueError("onset and duration must be non-negative")
        if self.protocol == "initial_state" and self.mode == "scale":
            raise ValueError("scale is not defined for an initial-state-only probe")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "protocol": self.protocol,
            "mode": self.mode,
            "amplitude": float(self.amplitude),
            "onset": int(self.onset),
            "duration": int(self.duration),
        }


@dataclass
class Rollout:
    """One free-running trajectory and its diagnostics."""

    states: np.ndarray  # (steps + 1, n_neurons)
    behavior_proba: np.ndarray  # (steps, 4)
    support_distance: np.ndarray  # (steps + 1,)
    seed: int
    model_kind: str
    noise_std: float
    perturbation: PerturbationSpec | None = None
    latent: np.ndarray | None = None
    valid: bool = True
    initial_history: np.ndarray | None = None
    native_stochastic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float64)
        self.behavior_proba = np.asarray(self.behavior_proba, dtype=np.float64)
        self.support_distance = np.asarray(self.support_distance, dtype=np.float64)
        if self.states.ndim != 2:
            raise ValueError("states must be (steps+1, n_neurons)")
        if self.behavior_proba.shape[0] != self.states.shape[0] - 1:
            raise ValueError("behavior_proba must have one row per transition")

    @property
    def steps(self) -> int:
        return int(self.states.shape[0] - 1)

    @property
    def ood_mask(self) -> np.ndarray:
        threshold = self.metadata.get("support_threshold")
        if threshold is None:
            return np.zeros(len(self.support_distance), dtype=bool)
        return self.support_distance > float(threshold)

    @property
    def ood_fraction(self) -> float:
        return float(np.mean(self.ood_mask)) if len(self.ood_mask) else 0.0

    @property
    def finite(self) -> bool:
        return bool(np.isfinite(self.states).all() and np.isfinite(self.behavior_proba).all())

    def predicted_macro(self) -> np.ndarray:
        return self.behavior_proba.argmax(axis=1)

    def to_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            states=self.states,
            behavior_proba=self.behavior_proba,
            support_distance=self.support_distance,
            latent=self.latent if self.latent is not None else np.empty((0, 0)),
            seed=np.asarray(self.seed),
            model_kind=np.asarray(self.model_kind),
            noise_std=np.asarray(self.noise_std),
            perturbation=np.asarray(str(self.perturbation.to_dict() if self.perturbation else "")),
        )


@dataclass
class RolloutEnsemble:
    """Several stochastic futures from one initial state."""

    x0: np.ndarray
    rollouts: tuple[Rollout, ...]
    seeds: tuple[int, ...]
    model_kind: str
    initial_macro: int | None = None

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=np.float64)
        if len(self.rollouts) != len(self.seeds):
            raise ValueError("rollouts and seeds must have equal length")

    @property
    def stack(self) -> np.ndarray:
        return np.stack([r.states for r in self.rollouts])

    @property
    def behavior_proba(self) -> np.ndarray:
        return np.stack([r.behavior_proba for r in self.rollouts])

    @property
    def ood_fraction(self) -> float:
        if not self.rollouts:
            return float("nan")
        return float(np.mean([r.ood_fraction for r in self.rollouts]))

    @property
    def stochastic(self) -> bool:
        """Whether the ensemble actually contains more than one trajectory."""
        if len(self.rollouts) < 2:
            return False
        return not np.allclose(self.stack, self.stack[:1], atol=1e-12)

    def outcome_distribution(self, horizon: int | None = None) -> np.ndarray:
        """Distribution of first macro state different from the initial state."""
        from neurobranch.data.kato import MACRO_ORDER

        initial = (
            int(self.initial_macro)
            if self.initial_macro is not None
            else (int(np.argmax(self.rollouts[0].behavior_proba[0])) if self.rollouts else 0)
        )
        counts = np.zeros(4, dtype=np.float64)
        for rollout in self.rollouts:
            probs = rollout.behavior_proba[: horizon if horizon is not None else rollout.steps]
            labels = probs.argmax(axis=1)
            outcome = initial
            for label in labels:
                if int(label) != initial:
                    outcome = int(label)
                    break
            counts[outcome] += 1.0
        del MACRO_ORDER
        return counts / max(1.0, counts.sum())


@dataclass(frozen=True)
class SimulationManifest:
    """Portable description of a simulation experiment."""

    schema_version: int
    git_revision: str
    git_dirty: bool
    data_path: str
    data_sha256: str
    protocol_path: str
    protocol_sha256: str
    model_kind: str
    checkpoint: str
    worm_id: int
    seed: int
    horizon: int
    noise_std: float
    perturbation: dict[str, Any] | None
    created_utc: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "git_revision": self.git_revision,
            "git_dirty": bool(self.git_dirty),
            "data": {"path": self.data_path, "sha256": self.data_sha256},
            "protocol": {"path": self.protocol_path, "sha256": self.protocol_sha256},
            "model_kind": self.model_kind,
            "checkpoint": self.checkpoint,
            "worm_id": int(self.worm_id),
            "seed": int(self.seed),
            "horizon": int(self.horizon),
            "noise_std": float(self.noise_std),
            "perturbation": self.perturbation,
            "created_utc": self.created_utc,
            "notes": list(self.notes),
        }
