"""The enumerating reference must not be able to call the simulator."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from neurobranch.abstraction import reference
from neurobranch.abstraction.systems import DiscreteFixture, discrete_fixtures


def test_reference_source_does_not_call_simulate() -> None:
    source = inspect.getsource(reference)
    assert ".simulate(" not in source
    assert "fixture.simulate" not in source


def test_exact_commutation_works_with_the_simulator_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("simulate called")

    monkeypatch.setattr(DiscreteFixture, "simulate", forbidden)
    fixtures = discrete_fixtures()
    results = reference.exact_commutation(fixtures["exactly_lumpable/positive"])
    assert results["passive"].dynamic == 0.0


def test_declared_dependencies_are_the_only_imports() -> None:
    assert "neurobranch.abstraction.estimands" in reference.DEPENDENCIES
    assert all("systems" in name or "estimands" in name or name == "numpy" for name in reference.DEPENDENCIES)


def test_enumeration_matches_a_hand_computed_case() -> None:
    fixture = discrete_fixtures()["exactly_lumpable/intervention_only"]
    lesion = reference.exact_commutation(fixture)["lesion"]
    # Block 1 holds states 2 and 3; only state 2 is re-targeted by the lesion.
    projection = fixture.macro_projection("lesion")
    expected = 0.5 * np.abs(projection[2] - projection[3]).sum()
    assert lesion.dynamic == pytest.approx(expected)
    assert expected > 0.05


def test_identifiability_report_separates_known_and_unknown_operators() -> None:
    fixture = discrete_fixtures()["ring_witness/L2"]
    report = reference.identifiability_report(fixture)
    assert report["identifiable_operators"] == ["passive"]
    assert report["unidentifiable_operators"] == ["asym"]
    assert report["fixture"] == "ring_witness/L2"
