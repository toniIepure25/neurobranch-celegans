"""Tests for design-resolution and equivalence utilities."""

from __future__ import annotations

import pytest

from neurobranch.benchmark.power import (
    classify_bound,
    classify_superiority,
    mde_from_ci,
    mde_two_arm,
    power_two_arm,
    sample_size_two_arm,
    tost,
)


def test_mde_matches_normal_approximation() -> None:
    value = mde_two_arm(20, 1.0)
    assert value == pytest.approx(0.885972, rel=1e-4)


def test_power_at_mde_is_target_power() -> None:
    mde = mde_two_arm(20, 1.0, power=0.8)
    assert power_two_arm(20, 1.0, mde) == pytest.approx(0.8, abs=1e-3)


def test_sample_size_inverts_mde() -> None:
    n = sample_size_two_arm(sd=1.0, effect=0.9, power=0.8)
    assert mde_two_arm(n, 1.0) <= 0.9


def test_mde_from_ci_matches_symmetric_interval() -> None:
    mde = mde_from_ci(0.0080, 0.0901)
    assert mde == pytest.approx(2.801585 * (0.0901 - 0.0080) / 3.919928, rel=1e-3)


def test_tost_detects_equivalence_and_non_equivalence() -> None:
    equivalent = tost(0.0, 0.01, margin=0.05)
    assert equivalent.equivalent
    assert equivalent.p_max < 1e-6
    different = tost(0.1, 0.01, margin=0.05)
    assert not different.equivalent


def test_classify_superiority_underpowered_when_mde_exceeds_margin() -> None:
    status = classify_superiority(0.0442, 0.0080, 0.0901, margin=0.02)
    assert status == "underpowered"


def test_classify_superiority_supported() -> None:
    status = classify_superiority(0.30, 0.20, 0.40, margin=0.20)
    assert status == "supported"


def test_classify_superiority_bounded_null() -> None:
    status = classify_superiority(0.005, -0.01, 0.02, margin=0.05)
    assert status == "bounded_null"


def test_classify_bound() -> None:
    assert classify_bound(0.22, 0.19, 0.25, bound=0.5, direction="below") == "supported"
    assert classify_bound(0.22, 0.19, 0.55, bound=0.5, direction="below") == "not_supported"
    assert classify_bound(0.8, 0.7, 0.9, bound=0.5, direction="above") == "supported"


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        mde_two_arm(1, 1.0)
    with pytest.raises(ValueError):
        tost(0.0, 0.0, 0.1)
    with pytest.raises(ValueError):
        classify_superiority(0.0, -1.0, 1.0, margin=0.0)
