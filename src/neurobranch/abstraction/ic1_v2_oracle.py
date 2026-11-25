"""Finite kernel propagation and enumeration for the IC1-v2 estimands."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from neurobranch.abstraction.ic1_v2_contract import InterventionDesign


@dataclass(frozen=True)
class OracleCell:
    starts: tuple[int, ...]
    passive_mass: float
    diameter: float
    weighted: float
    response_mixture: tuple[float, ...]


@dataclass(frozen=True)
class OracleResult:
    cells: dict[int, OracleCell]
    diameter: float
    weighted: float
    response_by_start: dict[int, tuple[float, ...]]


def enumerate_response_laws(design: InterventionDesign) -> OracleResult:
    """Propagate each positive passive start without restricting any later states."""
    by_cell: dict[int, list[int]] = {}
    support = []
    for state, (cell, weight) in enumerate(zip(design.fixture.partition, design.passive_weights, strict=True)):
        if weight > 0:
            support.append(state)
            by_cell.setdefault(int(cell), []).append(state)

    flow = np.eye(design.fixture.n_states)[support]
    for operator in design.operators:
        flow = flow @ design.fixture.kernel(operator)
    terminal_rows = dict(zip(support, flow @ design.emission, strict=True))
    total_weight = sum(design.passive_weights)

    cells = {}
    for cell, members in by_cell.items():
        cell_weight = sum(design.passive_weights[state] for state in members)
        conditional = {state: design.passive_weights[state] / cell_weight for state in members}
        mixture = sum(conditional[state] * terminal_rows[state] for state in members)
        diameter = weighted = 0.0
        for first, second in combinations(members, 2):
            distance = float(np.abs(terminal_rows[first] - terminal_rows[second]).sum() / 2)
            diameter = max(diameter, distance)
            weighted += 2 * conditional[first] * conditional[second] * distance
        cells[cell] = OracleCell(tuple(members), cell_weight / total_weight, diameter, weighted,
                                 tuple(float(value) for value in mixture))
    return OracleResult(
        cells,
        max(cell.diameter for cell in cells.values()),
        sum(cell.passive_mass * cell.weighted for cell in cells.values()),
        {state: tuple(float(value) for value in row) for state, row in terminal_rows.items()},
    )
