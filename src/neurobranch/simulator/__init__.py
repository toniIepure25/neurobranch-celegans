"""Free-running neural population simulator for NeuroBranch.

The public entry point is :class:`NeuralSimulator`.  It wraps an existing
fitted dynamics checkpoint and adds explicit rollout, support, perturbation and
manifest semantics.  A simulated trajectory is never an experimental
intervention.
"""

from __future__ import annotations

from neurobranch.simulator.adapters import NeuralSimulator, SupportModel
from neurobranch.simulator.types import (
    WARNING,
    PerturbationSpec,
    Rollout,
    RolloutEnsemble,
    SimulationManifest,
    SimulatorSpec,
)

__all__ = [
    "NeuralSimulator",
    "SupportModel",
    "WARNING",
    "PerturbationSpec",
    "Rollout",
    "RolloutEnsemble",
    "SimulationManifest",
    "SimulatorSpec",
]
