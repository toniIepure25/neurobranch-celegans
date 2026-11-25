"""Rational finite-ring oracle, independent of episode generation and fitting."""

from collections import defaultdict
from fractions import Fraction
from itertools import product

POS = range(3)
VELOCITIES = (-1, 1)
CONTEXTS = (0, 1)
CONTRACTS = ("full_base", "compressed_base", "full_sensor", "compressed_sensor")
SUPPORT = tuple(product(POS, VELOCITIES, CONTEXTS))


def response_law(position: int, velocity: int, context: int) -> tuple[Fraction, ...]:
    """Enumerate the two terminal position probabilities of the single probe."""
    law = [Fraction(0)] * 3
    law[(position + context + velocity) % 3] = Fraction(17, 20)
    law[(position + context - velocity) % 3] = Fraction(3, 20)
    return tuple(law)


def information_cell(state: tuple[int, int, int], contract: str) -> tuple[int, ...]:
    """Reconstruct the passive prehistory from the checkpoint for enumeration."""
    position, velocity, context = state
    match contract:
        case "full_base":
            return ((position - velocity) % 3, position)
        case "compressed_base":
            return (position,)
        case "full_sensor":
            return ((position - velocity) % 3, position, context)
        case "compressed_sensor":
            return (position, context)
    raise ValueError(f"unknown contract {contract!r}")


def tv(left: tuple, right: tuple) -> Fraction:
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / 2


def _bayes_action(laws: list[tuple[Fraction, ...]]) -> tuple[Fraction, ...]:
    """Minimize equally weighted L1 over the simplex using convex linear segments."""
    if len(laws) <= 2:
        return tuple(sum(values) / len(laws) for values in zip(*laws, strict=True))
    segments = []
    for category, values in enumerate(zip(*laws, strict=True)):
        breaks = sorted(set((Fraction(0), Fraction(1), *values)))
        for lower, upper in zip(breaks[:-1], breaks[1:], strict=True):
            slope = sum(1 if value <= lower else -1 for value in values)
            segments.append((slope, category, lower, upper - lower))
    action = [Fraction(0)] * len(laws[0])
    remaining = Fraction(1)
    for _, category, _, length in sorted(segments):
        added = min(length, remaining)
        action[category] += added
        remaining -= added
        if remaining == 0:
            break
    return tuple(action)


def oracle_tables(contract: str) -> tuple[dict, dict]:
    """Return exact Bayes actions and the conditional mean response laws."""
    by_cell = defaultdict(list)
    for state in SUPPORT:
        by_cell[information_cell(state, contract)].append(response_law(*state))
    optimal = {cell: _bayes_action(laws) for cell, laws in by_cell.items()}
    mixture = {cell: tuple(sum(values) / len(laws) for values in zip(*laws, strict=True))
               for cell, laws in by_cell.items()}
    return optimal, mixture


def mechanism_risks(predict, contract: str) -> tuple[Fraction, ...]:
    """Take the history/velocity expectation separately within each fixed mechanism."""
    return tuple(
        sum(tv(predict(information_cell(state, contract)), response_law(*state))
            for state in SUPPORT if state[2] == context) / 6
        for context in CONTEXTS
    )


def oracle_summary() -> dict:
    """Verify exact Bayes optima and the feasible midpoint minimax certificates."""
    summary = {}
    for contract in CONTRACTS:
        optimal, mean_response = oracle_tables(contract)
        bayes = sum(mechanism_risks(optimal.__getitem__, contract)) / 2
        witness = optimal
        if contract == "compressed_base":
            # A rotation of (.075, .075, .85) balances both mechanism risks.
            witness = {cell: tuple(Fraction(17, 20) if (y - cell[0]) % 3 == 2 else Fraction(3, 40)
                                   for y in POS) for cell in optimal}
        minimax_risks = mechanism_risks(witness.__getitem__, contract)
        if minimax_risks != (bayes, bayes):
            raise ValueError("minimax witness does not attain the Bayes lower bound in both mechanisms")
        mixture = sum(mechanism_risks(mean_response.__getitem__, contract)) / 2
        summary[contract] = {
            "bayes_optimum_exact": str(bayes),
            "minimax_optimum_exact": str(max(minimax_risks)),
            "minimax_witness_risks_exact": [str(value) for value in minimax_risks],
            "conditional_mean_risk_exact": str(mixture),
            "objective_gap_exact": str(mixture - bayes),
        }
    return summary
