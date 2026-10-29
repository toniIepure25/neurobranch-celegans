"""Design calculations for metric-aligned reliability goals."""

from neurobranch.design.calibration import coverage_study, simulate_atlas, true_reliability
from neurobranch.design.ceiling import (
    DEFAULT_GOALS,
    animals_for_reliability,
    design_table,
    required_animals_by_target,
)

__all__ = [
    "DEFAULT_GOALS",
    "animals_for_reliability",
    "coverage_study",
    "design_table",
    "required_animals_by_target",
    "simulate_atlas",
    "true_reliability",
]
