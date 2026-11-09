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

from .dataset import BEHAVIOR_NAMES, WormTrajectory, load_trajectories
from .dynamics import BehaviorReadout, PolynomialRidgeDynamics, fit_models

__all__ = [
    "BEHAVIOR_NAMES",
    "BehaviorReadout",
    "PolynomialRidgeDynamics",
    "WormTrajectory",
    "fit_models",
    "load_trajectories",
]
