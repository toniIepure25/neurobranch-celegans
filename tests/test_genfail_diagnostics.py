"""Lightweight invariants for the generative-failure diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from neurobranch.diagnostics.common import skill_vs_persistence
from neurobranch.diagnostics.oxygen import session_id
from neurobranch.diagnostics.perturbation import WARNING


def test_skill_is_zero_when_mse_matches_persistence() -> None:
    assert skill_vs_persistence(0.4, 0.4) == 0.0
    assert skill_vs_persistence(0.2, 0.4) == 0.5
    assert skill_vs_persistence(0.8, 0.4) == -1.0


def test_skill_is_nan_for_invalid_reference() -> None:
    assert np.isnan(skill_vs_persistence(0.2, 0.0))
    assert np.isnan(skill_vs_persistence(np.nan, 0.4))


def test_external_recording_has_stable_display_id() -> None:
    data = SimpleNamespace(worm_id=3)
    assert session_id(data) == "kato_stim_3"


def test_virtual_perturbation_warning_is_exact() -> None:
    assert WARNING == "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."
