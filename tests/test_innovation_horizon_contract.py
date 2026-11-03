"""Regression contract for the 20-step versus 100-step innovation error."""

from __future__ import annotations

import numpy as np

from neurobranch.probabilistic.stochastic import Innovation


def test_moving_block_draw_is_not_horizon_invariant() -> None:
    """A 20-step block draw is not a prefix of a 100-step draw.

    The moving-block sampler draws ``ceil(horizon / block)`` starts per member.
    A reference that asks for 20 innovations therefore receives different
    values at steps 1-20 than the official 100-step rollout. This is the root
    cause of the historical 1.473899 / 2.410819 reconstruction error.
    """
    pool = np.arange(2_000, dtype=np.float64).reshape(500, 4)
    innovation = Innovation("block", pool=pool, block=10)
    short = innovation.sample(8, 20, np.random.default_rng(7000))
    long = innovation.sample(8, 100, np.random.default_rng(7000))
    assert short.shape == (8, 20, 4)
    assert long.shape == (8, 100, 4)
    assert not np.allclose(short, long[:, :20])


def test_moving_block_draw_repeats_for_the_same_horizon() -> None:
    pool = np.arange(2_000, dtype=np.float64).reshape(500, 4)
    innovation = Innovation("block", pool=pool, block=10)
    first = innovation.sample(8, 100, np.random.default_rng(7000))
    second = innovation.sample(8, 100, np.random.default_rng(7000))
    np.testing.assert_array_equal(first, second)
