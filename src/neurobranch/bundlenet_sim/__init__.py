"""Three-tier Vienna Neuroinformatics workbench.

The package name is retained from the first spike, but the scope is now broader:

* Tier 1 — BunDLe-Net latent-manifold simulator (``dataset``, ``dynamics``,
  ``evaluate``, ``manifold_walk``).
* Tier 2 — NC-MCM causal-state reader (``ncmcm``).
* Tier 3 — biophysical premotor sandbox (``premotor``).
* ``workbench`` is the unified facade; ``app`` is the Streamlit UI.

Tier 1 models the *learned latent manifold*, not a biophysical or mechanistic
whole-brain system.  Tier 2 interventions are observational read-outs.  Tier 3
is a sandbox whose public export lacks gap-junction weights and neuron names.
"""

from .clamped import (
    ClampedParameters,
    ClampedResult,
    ClampedSimulationError,
    SignalPulse,
    simulate_clamped,
)
from .dataset import BEHAVIOR_NAMES, WormTrajectory, load_trajectories
from .dynamics import BehaviorReadout, PolynomialRidgeDynamics, fit_models
from .recording import (
    ClampedDataError,
    ClampedRecording,
    build_clamped_recording,
    load_atanas_recording,
)

__all__ = [
    "BEHAVIOR_NAMES",
    "BehaviorReadout",
    "ClampedDataError",
    "ClampedParameters",
    "ClampedRecording",
    "ClampedResult",
    "ClampedSimulationError",
    "PolynomialRidgeDynamics",
    "SignalPulse",
    "WormTrajectory",
    "build_clamped_recording",
    "fit_models",
    "load_atanas_recording",
    "load_trajectories",
    "simulate_clamped",
]
