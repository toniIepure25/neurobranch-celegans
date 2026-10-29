"""Design resolution, power and equivalence for cluster-level claims.

The Phase 6-8 reports compared point estimates and confidence intervals but
never asked whether the *design* could resolve the effect size the project
actually cares about.  This module makes that question explicit:

* :func:`mde_two_arm` -- smallest detectable effect for an animal-randomised
  two-arm design.
* :func:`mde_from_ci` -- the same quantity reconstructed from a reported 95%
  interval, for retrospective audit of an existing result.
* :func:`tost` -- two one-sided tests for an equivalence margin.
* :func:`classify_superiority` / :func:`classify_bound` -- a fixed decision
  table mapping estimate, interval, margin and MDE to a claim status.

All functions are normal-approximation.  They assume independent animal-level
units; a design that reuses animals across arms needs a paired or blocked MDE
instead.  Nothing here turns an observational contrast into a causal effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from statistics import NormalDist
from typing import Literal

ClaimStatus = Literal["supported", "not_supported", "bounded_null", "underpowered"]

_NORMAL = NormalDist()


def _z(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    return _NORMAL.inv_cdf(probability)


def mde_two_arm(
    n_per_arm: int,
    sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Minimum detectable effect for a two-arm design with cluster means.

    ``sd`` is the between-animal standard deviation of the endpoint.  The
    return value is on the same scale as the endpoint.
    """
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be at least 2")
    if sd <= 0:
        raise ValueError("sd must be positive")
    return (_z(1 - alpha / 2) + _z(power)) * sd * sqrt(2.0 / n_per_arm)


def power_two_arm(
    n_per_arm: int,
    sd: float,
    effect: float,
    *,
    alpha: float = 0.05,
) -> float:
    """Approximate two-sided power for a two-arm design."""
    if n_per_arm < 2 or sd <= 0:
        raise ValueError("n_per_arm must be at least 2 and sd positive")
    standard_error = sd * sqrt(2.0 / n_per_arm)
    noncentrality = abs(effect) / standard_error
    critical = _z(1 - alpha / 2)
    return 1.0 - _NORMAL.cdf(critical - noncentrality) + _NORMAL.cdf(-critical - noncentrality)


def sample_size_two_arm(
    sd: float,
    effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Animals per arm required to detect ``effect`` with the given power."""
    if sd <= 0 or effect == 0:
        raise ValueError("sd must be positive and effect non-zero")
    n = 2.0 * (((_z(1 - alpha / 2) + _z(power)) * sd) / abs(effect)) ** 2
    return max(2, ceil(n))


def mde_from_ci(
    ci_low: float,
    ci_high: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Reconstruct the design's MDE from a reported two-sided 95% interval.

    This is a retrospective resolution check, not a post-hoc power calculation
    for the observed effect.  It answers: "what is the smallest effect this
    design could have detected at the stated power?"
    """
    if not ci_high > ci_low:
        raise ValueError("ci_high must exceed ci_low")
    standard_error = (ci_high - ci_low) / (2.0 * _z(1 - alpha / 2))
    return (_z(1 - alpha / 2) + _z(power)) * standard_error


@dataclass(frozen=True)
class TostResult:
    """Two one-sided tests against an equivalence margin."""

    estimate: float
    standard_error: float
    margin: float
    p_lower: float
    p_upper: float
    p_max: float
    equivalent: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "margin": self.margin,
            "p_lower": self.p_lower,
            "p_upper": self.p_upper,
            "p_max": self.p_max,
            "equivalent": self.equivalent,
        }


def tost(
    estimate: float,
    standard_error: float,
    margin: float,
    *,
    alpha: float = 0.05,
) -> TostResult:
    """Two one-sided tests for ``|effect| < margin``.

    ``p_max < alpha`` is the conventional equivalence decision.  A confidence
    interval that lies entirely inside ``(-margin, margin)`` is equivalent to
    the same decision at the matching level.
    """
    if standard_error <= 0:
        raise ValueError("standard_error must be positive")
    if margin <= 0:
        raise ValueError("margin must be positive")
    z_lower = (estimate + margin) / standard_error
    z_upper = (estimate - margin) / standard_error
    p_lower = 1.0 - _NORMAL.cdf(z_lower)
    p_upper = _NORMAL.cdf(z_upper)
    p_max = max(p_lower, p_upper)
    return TostResult(
        estimate=estimate,
        standard_error=standard_error,
        margin=margin,
        p_lower=p_lower,
        p_upper=p_upper,
        p_max=p_max,
        equivalent=bool(p_max < alpha),
    )


def classify_superiority(
    estimate: float,
    ci_low: float,
    ci_high: float,
    *,
    margin: float,
    mde: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
) -> ClaimStatus:
    """Classify a directional "A beats B" claim.

    Decision order matters and is preregistered here:

    1. ``underpowered`` if the design MDE exceeds the smallest effect of
       interest (``margin``); the design cannot resolve what matters.
    2. ``supported`` if the interval excludes zero and the point estimate
       reaches the MDE.
    3. ``bounded_null`` if the interval lies inside ``(-margin, margin)``.
    4. ``not_supported`` otherwise.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    if mde is None:
        mde = mde_from_ci(ci_low, ci_high, alpha=alpha, power=power)
    if mde > margin:
        return "underpowered"
    excludes_zero = ci_low > 0.0 or ci_high < 0.0
    if excludes_zero and abs(estimate) >= mde:
        return "supported"
    if -margin < ci_low and ci_high < margin:
        return "bounded_null"
    return "not_supported"


def classify_bound(
    estimate: float,
    ci_low: float,
    ci_high: float,
    *,
    bound: float,
    direction: Literal["below", "above"],
) -> ClaimStatus:
    """Classify a one-sided bound claim such as "reliability is below 0.5"."""
    if direction == "below":
        return "supported" if ci_high < bound else "not_supported"
    return "supported" if ci_low > bound else "not_supported"


__all__ = [
    "ClaimStatus",
    "TostResult",
    "classify_bound",
    "classify_superiority",
    "mde_from_ci",
    "mde_two_arm",
    "power_two_arm",
    "sample_size_two_arm",
    "tost",
]
