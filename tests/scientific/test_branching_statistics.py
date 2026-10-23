from __future__ import annotations

import numpy as np

from neurobranch.branching.stats import (
    block_permute,
    local_entropies,
    moving_block_bootstrap_mean,
    null_branching_test,
)
from neurobranch.branching.synthetic import make_scenario, outcomes_for


def test_deterministic_outcome_has_zero_entropy():
    s = make_scenario("deterministic", n_candidates=40, seed=1)
    e = local_entropies(s.Z, s.candidate_frames, outcomes_for(s), k=15)
    assert np.allclose(e, 0.0)


def test_stochastic_branch_has_high_entropy():
    s = make_scenario("stochastic", n_candidates=60, seed=2)
    e = local_entropies(s.Z, s.candidate_frames, outcomes_for(s), k=15)
    assert e.mean() > 0.8


def test_rare_transition_is_low_but_nonzero_entropy():
    s = make_scenario("rare", n_candidates=60, seed=3)
    e = local_entropies(s.Z, s.candidate_frames, outcomes_for(s), k=15)
    assert 0.05 < e.mean() < 0.45


def test_projection_can_create_apparent_branching():
    """A 3-D projection can hide two well-separated high-D states."""
    s = make_scenario("projection", n_candidates=60, seed=4)
    y = outcomes_for(s)
    low_dim = local_entropies(s.Z, s.candidate_frames, y, k=10)
    high_dim = local_entropies(s.Z_high, s.candidate_frames, y, k=10)
    assert low_dim.mean() > 0.7
    assert high_dim.mean() < 0.1


def test_hidden_state_branching_is_real_but_unobserved():
    s = make_scenario("hidden", n_candidates=60, seed=5)
    y = outcomes_for(s)
    e = local_entropies(s.Z, s.candidate_frames, y, k=10)
    assert e.mean() > 0.5
    # Conditioning on the hidden state removes the entropy: this is
    # unobserved-state mixing, not stochastic dynamics at a fixed state.
    for h in (0, 1):
        mask = s.hidden_state == h
        sub = local_entropies(s.Z, s.candidate_frames[mask], y[mask], k=5)
        assert sub.mean() < 1e-9


def test_block_permutation_preserves_labels_but_reorders_blocks():
    rng = np.random.default_rng(0)
    x = np.repeat(np.arange(10), 5)
    y = block_permute(x, rng, block_size=5)
    assert sorted(y.tolist()) == sorted(x.tolist())
    assert not np.array_equal(x, y)


def test_directional_p_values_do_not_call_a_deficit_branching():
    # High-dimensional geometry separates the two outcome classes; the i.i.d.
    # null destroys that alignment and therefore has higher entropy.  This is a
    # deficit, not evidence for branching.
    s = make_scenario("projection", n_candidates=50, seed=6)
    out = null_branching_test(
        s.Z_high,
        s.candidate_frames,
        outcomes_for(s),
        k=10,
        n_permutations=100,
        method="iid",
        seed=0,
        bootstrap=100,
    )
    assert out["p_excess"] > 0.5
    assert out["p_deficit"] < 0.1
    assert out["p_two_sided"] <= 0.2
    assert out["excess_entropy_bits"] < 0


def test_block_bootstrap_interval_is_ordered_and_finite():
    x = np.linspace(0, 1, 100)
    out = moving_block_bootstrap_mean(x, block_size=10, n_bootstrap=200, seed=0)
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]
    assert np.isfinite([out["ci_low"], out["ci_high"]]).all()


def test_shift_null_is_not_identical_to_iid_null():
    s = make_scenario("stochastic", n_candidates=50, seed=7)
    y = outcomes_for(s)
    a = null_branching_test(
        s.Z, s.candidate_frames, y, k=15, n_permutations=40, method="iid", seed=0, bootstrap=50
    )
    b = null_branching_test(
        s.Z, s.candidate_frames, y, k=15, n_permutations=40, method="shift", seed=0, bootstrap=50
    )
    assert a["null_method"] == "iid"
    assert b["null_method"] == "shift"
    assert a["n_valid_null"] > 0 and b["n_valid_null"] > 0
