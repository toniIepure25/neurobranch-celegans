"""NeuroBranch: C. elegans population dynamics, branching and virtual perturbations.

The package is deliberately explicit about epistemic status.  Predictions from
models trained only on observational recordings are *model-dependent
counterfactuals*, not validated causal effects.
"""

from __future__ import annotations

__version__ = "1.1.0"

EPISTEMIC_WARNING = (
    "MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED. "
    "This output is a simulation under stated model assumptions, not evidence "
    "that the same perturbation controls a living animal."
)

__all__ = ["__version__", "EPISTEMIC_WARNING"]
