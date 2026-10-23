from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.interventions import InterventionSpec, run_counterfactual


def test_counterfactual_reports_epistemic_warning(toy_model, toy_data):
    spec = InterventionSpec(
        name="toy",
        targets=("RIML",),
        amplitude=1.0,
        horizon=4,
        n_initial=5,
        seeds=(0, 1),
    )
    r = run_counterfactual(toy_model, toy_data, spec, seed=0, noise_std=0.0)
    assert r.causal_status.validated is False
    assert "NOT CAUSALLY VALIDATED" in r.causal_status.label
    assert set(r.baseline_distribution) == {"FWD", "REV", "TURN", "NOSTATE"}
    assert np.isclose(sum(r.intervention_distribution.values()), 1.0)
    assert isinstance(r.per_initial, pd.DataFrame)


def test_counterfactual_is_paired(toy_model, toy_data):
    spec = InterventionSpec(name="noop", targets=("RIML",), amplitude=0.0, horizon=4, n_initial=4, seeds=(0,))
    r = run_counterfactual(toy_model, toy_data, spec, seed=0, noise_std=0.0)
    assert abs(r.delta_target) < 1e-12
    assert r.mean_trajectory_displacement < 1e-12
