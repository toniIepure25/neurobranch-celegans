"""Kato-only neural interaction and sensitivity diagnostics.

This package deliberately contains one small linear family: lag-one VARs,
their diagonal (own-neuron-only) ablations, and matched temporal surrogates.
Nothing here is a biological intervention model.
"""

from neurobranch.kato_interactions.models import FrozenLinearModel, fit_window, fit_window_diagonal
from neurobranch.kato_interactions.surrogates import fit_shuffled_window

__all__ = [
    "FrozenLinearModel",
    "fit_shuffled_window",
    "fit_window",
    "fit_window_diagonal",
]
