"""Independent enumerating reference for the interventional-consistency estimands.

This module is the *oracle* path.  It reads only the fixture's conditional arrays
(``kernels``, ``partition``, ``behavior``) and never calls ``simulate``.  The
sampling estimator in ``commutation.py`` is the other path and shares no code with
it beyond the pure distance functions in ``estimands``.  Gate G1 of the frozen
protocol requires the two to agree to ``1e-9`` on an exactly lumpable fixture, which
is only a meaningful check if the paths are genuinely separate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from neurobranch.abstraction.estimands import CommutationError, commutation_from_conditional
from neurobranch.abstraction.systems import DiscreteFixture

# Modules this reference is allowed to depend on.  ``systems`` provides the fixture
# arrays; ``simulate`` is deliberately *not* used anywhere in this module.
DEPENDENCIES: tuple[str, ...] = ("numpy", "neurobranch.abstraction.estimands", "neurobranch.abstraction.systems")

LUMPABILITY_ATOL = 1e-12


def exact_commutation(
    fixture: DiscreteFixture,
    *,
    min_cell_count: int = 2,
    weight_by_occupancy: bool = True,
    occupancy_floor: float = 1e-6,
) -> dict[str, CommutationError]:
    """Sup-based commutation error for every operator, computed by enumeration.

    ``occupancy_floor`` excludes macro-cell members that the dynamics cannot reach
    with more than a millionth of the cell's peak occupancy.  Those members are
    artefacts of the product state space (a latent value that is incompatible with
    the observation window), and including them would make the error independent of
    the window length by construction.

    Occupancy is always measured under the **passive** kernel, even when the error is
    computed for another operator.  The abstraction is formed in the reference regime,
    so that is where "can this member of the cell actually occur" has to be decided;
    using the intervened kernel's stationary distribution would let an operator that
    makes some contexts impossible silently shrink its own error, which is the exact
    failure the witness fixture is built to expose.
    """
    results: dict[str, CommutationError] = {}
    reference_operator = "passive" if "passive" in fixture.operators else fixture.operators[0]
    occupancy = fixture.stationary_weights(reference_operator) if weight_by_occupancy else None
    for operator in fixture.operators:
        results[operator] = commutation_from_conditional(
            operator=operator,
            macro_by_state=fixture.macro_projection(operator),
            behavior_by_state=fixture.behavior,
            partition=fixture.partition,
            state_counts=occupancy,
            min_cell_count=min_cell_count,
            occupancy_floor=occupancy_floor,
        )
    return results


def is_exactly_lumpable(fixture: DiscreteFixture, operator: str, *, atol: float = LUMPABILITY_ATOL) -> bool:
    """True when every macro cell has identical next-cell rows across its members."""
    projection = fixture.macro_projection(operator)
    blocks = np.asarray(fixture.partition, dtype=int)
    for cell in np.unique(blocks):
        members = np.flatnonzero(blocks == cell)
        rows = projection[members]
        if not np.allclose(rows, rows[0], atol=atol):
            return False
    return True


def macro_transition_of_cell(
    fixture: DiscreteFixture,
    operator: str,
    cell: int,
    *,
    weight: str = "occupancy",
) -> np.ndarray:
    """``P_C^{omega(u)}(next cell | cell)`` for one starting cell.

    ``occupancy`` weights members by their stationary probability inside the cell,
    which is the only weighting that makes the macro chain well defined without
    assuming the cell is a sufficient statistic.
    """
    projection = fixture.macro_projection(operator)
    blocks = np.asarray(fixture.partition, dtype=int)
    members = np.flatnonzero(blocks == cell)
    if members.size == 0:
        raise ValueError(f"cell {cell} has no members")
    weights = fixture.stationary_weights(operator)[members]
    if weight != "occupancy":
        raise ValueError(f"unsupported weighting {weight!r}")
    total = weights.sum()
    weights = np.full(members.size, 1.0 / members.size) if total <= 0 else weights / total
    return weights @ projection[members]


def identifiability_report(fixture: DiscreteFixture) -> dict[str, Any]:
    """Which operators can be expressed at the macro level at all."""
    return {
        "fixture": fixture.name,
        "operators": list(fixture.operators),
        "identifiable_operators": list(fixture.identifiable_operators),
        "unidentifiable_operators": [
            operator for operator in fixture.operators if operator not in fixture.identifiable_operators
        ],
        "note": (
            "An operator is unidentifiable when its effect depends on a micro variable that "
            "no permitted history can separate; the report must then say not_identified "
            "instead of quoting a commutation error as if it were a property of the abstraction."
        ),
    }


def exact_summary(fixtures: Mapping[str, DiscreteFixture]) -> dict[str, Any]:
    """Compact exact summary for the conformance report."""
    summary: dict[str, Any] = {}
    for name, fixture in fixtures.items():
        summary[name] = {
            "fixture": fixture.as_dict(),
            "lumpable": {operator: is_exactly_lumpable(fixture, operator) for operator in fixture.operators},
            "errors": {operator: error.as_dict() for operator, error in exact_commutation(fixture).items()},
            "identifiability": identifiability_report(fixture),
        }
    return summary
