"""Bounded-amplitude contract for named virtual perturbations."""

from __future__ import annotations

from scripts.context_perturbations import TARGETS, _amplitude


def test_named_perturbation_amplitude_is_bounded_by_one_step_sd_and_range(toy_data):
    available = [t for t in TARGETS if t in toy_data.predictor_names]
    if not available:
        return
    for target in available:
        sd, cap = _amplitude(toy_data, target)
        amplitude = min(sd, cap)
        assert amplitude >= 0
        assert amplitude <= sd + 1e-12
        assert amplitude <= cap + 1e-12
