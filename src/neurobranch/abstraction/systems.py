"""Ground-truth systems for the interventional-consistency benchmark.

Three families, each with a positive control and two declared negatives:

``exactly_lumpable``
    A discrete 6-state chain with three macro blocks.  The positive fixture is
    exactly lumpable, so the macro process commutes with the micro process for
    every operator.  The negatives break exactly one property at a time:
    ``behaviour_only`` keeps the behavioural readout block-constant while the block
    transition probabilities differ within a block; ``intervention_only`` is
    exactly lumpable passively and breaks only under one operator.

``ring_state_space``
    A position on a three-site ring, moved by a latent velocity and observed as a
    deterministic symbol.  With ``L = 1`` the velocity is ambiguous inside each
    macro cell, so two members move to different symbols and the macro process does
    not commute.  A window of length two contains the displacement and fixes the
    velocity, so the error is exactly zero from ``L = 2``.  The witness adds a
    context variable that never touches the observation, plus an operator whose
    effect depends on that context: no history can separate the two contexts, the
    error stays at its ``L = 1`` value for every ``L``, and the operator must be
    reported as not identified.

``rate_circuit``
    A continuous three-population rate model with a hidden modulator, a softmax
    readout and Gaussian observation noise.  It is not enumerable; it exists so the
    sampling estimator is exercised on a system where the exact answer is unavailable.

Every discrete fixture exposes ``kernels`` (operator -> row-stochastic matrix),
``partition``, ``behavior`` and ``simulate``.  The enumerating reference in
``reference.py`` uses only the arrays; the simulator steps states.  They are
deliberately different code paths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neurobranch.abstraction.estimands import EstimandError, as_distribution


@dataclass(frozen=True)
class DiscreteFixture:
    """A finite micro system with declared operators and a coarse-graining."""

    name: str
    state_labels: tuple[str, ...]
    partition: np.ndarray
    behavior: np.ndarray
    kernels: Mapping[str, np.ndarray]
    operators: tuple[str, ...]
    identifiable_operators: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_states = len(self.state_labels)
        blocks = np.asarray(self.partition, dtype=int)
        if blocks.shape != (n_states,):
            raise EstimandError("partition must have one entry per state")
        if np.any(blocks < 0):
            raise EstimandError("partition ids must be non-negative")
        if self.behavior.shape != (n_states, self.n_behaviors):
            raise EstimandError("behavior must have shape (n_states, n_behaviors)")
        for operator in self.operators:
            if operator not in self.kernels:
                raise EstimandError(f"missing kernel for operator {operator!r}")
            kernel = np.asarray(self.kernels[operator], dtype=float)
            if kernel.shape != (n_states, n_states):
                raise EstimandError(f"kernel {operator!r} has shape {kernel.shape}, expected {(n_states, n_states)}")
            if np.any(kernel < -1e-12) or not np.all(np.isfinite(kernel)):
                raise EstimandError(f"kernel {operator!r} has negative or non-finite entries")
            if not np.allclose(kernel.sum(axis=1), 1.0, atol=1e-9):
                raise EstimandError(f"kernel {operator!r} rows do not sum to 1")
        unknown = set(self.identifiable_operators) - set(self.operators)
        if unknown:
            raise EstimandError(f"unknown identifiable operators: {sorted(unknown)}")

    @property
    def n_states(self) -> int:
        return len(self.state_labels)

    @property
    def n_behaviors(self) -> int:
        return int(np.asarray(self.behavior).shape[1])

    @property
    def n_cells(self) -> int:
        return int(np.unique(self.partition).size)

    def kernel(self, operator: str) -> np.ndarray:
        if operator not in self.kernels:
            raise EstimandError(f"unknown operator {operator!r}; have {sorted(self.kernels)}")
        return np.asarray(self.kernels[operator], dtype=float)

    def macro_projection(self, operator: str) -> np.ndarray:
        """``P(next macro cell | state, operator)`` for every micro state."""
        kernel = self.kernel(operator)
        cells = np.unique(self.partition)
        cells = np.atleast_1d(cells)
        index = np.searchsorted(cells, self.partition)
        if index.max() >= cells.size or np.any(cells[index] != self.partition):
            raise EstimandError("partition ids are not aligned with np.unique order")
        membership = np.zeros((self.n_states, cells.size), dtype=float)
        membership[np.arange(self.n_states), index] = 1.0
        return kernel @ membership

    def stationary_weights(self, operator: str, *, iterations: int = 500) -> np.ndarray:
        """Left eigenvector of the kernel, used to weight cells by occupancy."""
        kernel = self.kernel(operator)
        weights = np.full(self.n_states, 1.0 / self.n_states)
        for _ in range(iterations):
            updated = weights @ kernel
            updated = updated / updated.sum() if updated.sum() > 0 else weights
            if np.max(np.abs(updated - weights)) < 1e-12:
                weights = updated
                break
            weights = updated
        return weights / weights.sum()

    def simulate(
        self,
        operator: str,
        *,
        n_chains: int,
        n_steps: int,
        rng: np.random.Generator,
        initial: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Step chains through the kernel; returns states of shape (n_chains, n_steps + 1)."""
        kernel = self.kernel(operator)
        cumulative = np.cumsum(kernel, axis=1)
        if initial is None:
            states = rng.integers(0, self.n_states, size=n_chains)
        else:
            states = np.asarray(initial, dtype=int).copy()
            if states.shape != (n_chains,):
                raise EstimandError("initial must provide one state per chain")
        out = np.empty((n_chains, n_steps + 1), dtype=int)
        out[:, 0] = states
        for step in range(n_steps):
            draw = rng.random(n_chains)
            states = (draw[:, None] >= cumulative[states]).sum(axis=1)
            states = np.clip(states, 0, self.n_states - 1)
            out[:, step + 1] = states
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_states": self.n_states,
            "n_cells": self.n_cells,
            "n_behaviors": self.n_behaviors,
            "operators": list(self.operators),
            "identifiable_operators": list(self.identifiable_operators),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Family 1: exactly lumpable discrete chain
# ---------------------------------------------------------------------------

_BLOCK_KERNEL = np.array(
    [
        [0.70, 0.20, 0.10],
        [0.30, 0.50, 0.20],
        [0.20, 0.30, 0.50],
    ]
)
_BEHAVIOR_BY_BLOCK = np.array(
    [
        [0.80, 0.15, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
    ]
)


def _expand_to_states(block_rows: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """Repeat each block row across the micro states of that block."""
    rows = []
    for block, size in enumerate(sizes):
        rows.extend([block_rows[block]] * size)
    return np.vstack(rows)


def _block_states(sizes: Sequence[int]) -> np.ndarray:
    return np.repeat(np.arange(len(sizes)), sizes)


def _exactly_lumpable_positive() -> DiscreteFixture:
    sizes = (2, 2, 2)
    states = [f"s{index}" for index in range(sum(sizes))]
    blocks = _block_states(sizes)
    sizes_by_block = np.asarray(sizes, dtype=float)

    base = np.array([_BLOCK_KERNEL[block] for block in blocks])
    # Distribute each block row uniformly inside the destination block.
    kernel = np.zeros((len(states), len(states)))
    for state in range(len(states)):
        for target in range(len(states)):
            kernel[state, target] = base[state, blocks[target]] / sizes_by_block[blocks[target]]
    behavior = np.array([_BEHAVIOR_BY_BLOCK[block] for block in blocks])
    return DiscreteFixture(
        name="exactly_lumpable/positive",
        state_labels=tuple(states),
        partition=blocks,
        behavior=behavior,
        kernels={"passive": kernel},
        operators=("passive",),
        identifiable_operators=("passive",),
        metadata={"sizes": sizes, "role": "positive control"},
    )


def _exactly_lumpable_behaviour_only() -> DiscreteFixture:
    """Behaviour is block-constant; block transitions differ within a block."""
    positive = _exactly_lumpable_positive()
    kernel = positive.kernel("passive").copy()
    perturbation = 0.18
    # State 0 inside block 0 leaks mass towards block 2; state 1 towards block 1.
    kernel[0] = (1 - perturbation) * kernel[0] + perturbation * np.array(
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
    )
    kernel[1] = (1 - perturbation) * kernel[1] + perturbation * np.array(
        [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
    )
    return DiscreteFixture(
        name="exactly_lumpable/behaviour_only",
        state_labels=positive.state_labels,
        partition=positive.partition,
        behavior=positive.behavior,
        kernels={"passive": kernel},
        operators=("passive",),
        identifiable_operators=("passive",),
        metadata={"sizes": (2, 2, 2), "role": "negative control: dynamic aliasing, behaviour preserved"},
    )


def _exactly_lumpable_intervention_only() -> DiscreteFixture:
    """Exactly lumpable passively; one operator breaks the block structure."""
    positive = _exactly_lumpable_positive()
    passive = positive.kernel("passive")
    lesion = passive.copy()
    # A targeted perturbation that only reaches micro state 2 (block 1).
    lesion[2] = np.array([0.35, 0.35, 0.0, 0.0, 0.0, 0.30])
    return DiscreteFixture(
        name="exactly_lumpable/intervention_only",
        state_labels=positive.state_labels,
        partition=positive.partition,
        behavior=positive.behavior,
        kernels={"passive": passive, "lesion": lesion},
        operators=("passive", "lesion"),
        identifiable_operators=("passive", "lesion"),
        metadata={"sizes": (2, 2, 2), "role": "negative control: only an intervention exposes the failure"},
    )


# ---------------------------------------------------------------------------
# Family 2: latent ring with a deterministic symbol (history ladder + witness)
# ---------------------------------------------------------------------------

_N_POSITIONS = 3
_VELOCITIES = (1, -1)
_PERSISTENCE = 0.85
_BEHAVIOR_BY_POSITION = np.array(
    [
        [0.85, 0.15],
        [0.50, 0.50],
        [0.15, 0.85],
    ]
)


def _history_fixture(*, length: int, with_context: bool, name: str, role: str) -> DiscreteFixture:
    """Augmented chain over (position, velocity, context, window of ``length`` symbols).

    The observation is the position on a three-site ring, emitted deterministically,
    so a single symbol leaves the velocity ambiguous: two members of the same macro
    cell move to different next symbols and the macro process does not commute.
    A window of length two contains the displacement and therefore fixes the
    velocity, which makes every reachable cell a function of the state and drives the
    error to zero.

    The witness adds a context variable that never enters the observation, plus an
    operator whose effect on motion depends on that context.  Cells keep two
    reachable members for every window length, so the error does not shrink: the
    operator is not identified at the macro level and the report must say so.
    """
    n_contexts = 2 if with_context else 1
    n_windows = _N_POSITIONS**length
    n_states = _N_POSITIONS * len(_VELOCITIES) * n_contexts * n_windows

    def index(position: int, velocity_index: int, context: int, window: int) -> int:
        return ((position * len(_VELOCITIES) + velocity_index) * n_contexts + context) * n_windows + window

    def digits_of(window: int) -> list[int]:
        digits = [0] * length
        remainder = window
        for position in range(length - 1, -1, -1):
            digits[position] = remainder % _N_POSITIONS
            remainder //= _N_POSITIONS
        return digits

    state_labels: list[str] = []
    partition = np.empty(n_states, dtype=int)
    behavior = np.empty((n_states, _BEHAVIOR_BY_POSITION.shape[1]))
    for position in range(_N_POSITIONS):
        for velocity_index, velocity in enumerate(_VELOCITIES):
            for context in range(n_contexts):
                for window in range(n_windows):
                    state = index(position, velocity_index, context, window)
                    state_labels.append(f"p{position}v{velocity:+d}c{context}|h{digits_of(window)}")
                    partition[state] = window
                    behavior[state] = _BEHAVIOR_BY_POSITION[position]

    def build_kernel(operator: str) -> np.ndarray:
        if operator not in ("passive", "asym"):
            raise EstimandError(f"unknown operator {operator!r}")
        kernel = np.zeros((n_states, n_states))
        for position in range(_N_POSITIONS):
            for velocity_index in range(len(_VELOCITIES)):
                for context in range(n_contexts):
                    for next_index, probability in (
                        (velocity_index, _PERSISTENCE),
                        (1 - velocity_index, 1.0 - _PERSISTENCE),
                    ):
                        # Velocity is updated at the step boundary and the move then
                        # uses the new velocity.  That ordering is what lets a
                        # two-symbol window identify the *current* velocity; updating
                        # after the move would leave it one step behind, the window
                        # would never pin it down, and the error could not reach zero.
                        applied = _VELOCITIES[next_index]
                        kick = 0
                        if operator == "asym" and context == 1:
                            # The operator's effect depends on a context that never
                            # enters the observation.  A positional kick, rather than a
                            # velocity flip, is used on purpose: a flip would be an
                            # exact symmetry between (v, context) pairs and could look
                            # like a commutation that is merely relabelled.
                            kick = 1
                        next_position = (position + applied + kick) % _N_POSITIONS
                        for window in range(n_windows):
                            shifted = (window % (_N_POSITIONS ** (length - 1))) * _N_POSITIONS
                            state = index(position, velocity_index, context, window)
                            target = index(
                                next_position, next_index, context, shifted + next_position
                            )
                            kernel[state, target] += probability
        return kernel

    operators = ("passive", "asym") if with_context else ("passive",)
    return DiscreteFixture(
        name=name,
        state_labels=tuple(state_labels),
        partition=partition,
        behavior=behavior,
        kernels={operator: build_kernel(operator) for operator in operators},
        operators=operators,
        identifiable_operators=("passive",) if with_context else operators,
        metadata={
            "family": "ring_state_space",
            "history_length": length,
            "persistence": _PERSISTENCE,
            "role": role,
            "note": "deterministic position symbol on a three-site ring; displacement identifies velocity",
        },
    )


# ---------------------------------------------------------------------------
# Family 3: sampled rate circuit (not enumerable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateCircuit:
    """Continuous three-population rate model with a hidden modulator.

    ``x`` holds two visible populations; ``m`` is a hidden modulator that never
    enters the observation.  The macro cell is a 2x2 quantisation of the visible
    plane, declared before any run.
    """

    gain: float = 1.2
    tau: float = 0.35
    modulator_tau: float = 0.8
    noise: float = 0.05
    bins: int = 4
    extent: float = 1.2

    @property
    def n_macro(self) -> int:
        return self.bins * self.bins

    def sample_initial(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 0.5, size=(n, 3))

    def step(self, states: np.ndarray, operator: str, rng: np.random.Generator) -> np.ndarray:
        x = np.asarray(states, dtype=float)
        visible, modulator = x[:, :2], x[:, 2]
        # One draw for all sources of noise, so a fixed-noise generator can hold the
        # perturbations constant across start states (common random numbers).
        draws = rng.normal(0.0, 1.0, size=(x.shape[0], 3))
        drive = np.tanh(self.gain * visible + 0.15 * modulator[:, None])
        updated = visible + (self.tau * (-visible + drive)).clip(-0.6, 0.6)
        if operator == "lesion":
            updated[:, 0] = updated[:, 0] - 0.45
        elif operator == "clamp":
            updated[:, 0] = 0.6
        elif operator != "passive":
            raise EstimandError(f"unknown operator {operator!r}")
        updated_modulator = modulator + self.modulator_tau * (-modulator * 0.5) + self.noise * draws[:, 0]
        noise = self.noise * draws[:, 1:3]
        return np.column_stack([updated + noise, updated_modulator])

    def observe(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        visible = np.asarray(states, dtype=float)[:, :2]
        return visible + rng.normal(0.0, 0.08, size=visible.shape)

    def cell_of(self, observations: np.ndarray) -> np.ndarray:
        observed = np.asarray(observations, dtype=float)
        scaled = (observed + self.extent) / (2.0 * self.extent)
        digits = np.clip((scaled * self.bins).astype(int), 0, self.bins - 1)
        return digits[:, 0] * self.bins + digits[:, 1]

    def behavior_probs(self, states: np.ndarray) -> np.ndarray:
        visible = np.asarray(states, dtype=float)[:, :2].mean(axis=1)
        p_forward = 1.0 / (1.0 + np.exp(-2.0 * visible))
        return np.column_stack([p_forward, 1.0 - p_forward])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DISCRETE_WITNESS_OPERATORS = ("passive", "asym")

#: Window lengths enumerated exactly.  The augmented state space grows as
#: ``6 * contexts * 3**L``, so ``L = 8`` alone would need a 78k x 78k kernel.  The
#: exact grid therefore stops at 4; longer windows belong to a sampling estimator
#: and are out of scope for Stage One rather than silently approximated.
HISTORY_LENGTHS: tuple[int, ...] = (1, 2, 4)


def discrete_fixtures() -> dict[str, DiscreteFixture]:
    """Every enumerable fixture, keyed by a stable identifier."""
    fixtures = {
        "exactly_lumpable/positive": _exactly_lumpable_positive(),
        "exactly_lumpable/behaviour_only": _exactly_lumpable_behaviour_only(),
        "exactly_lumpable/intervention_only": _exactly_lumpable_intervention_only(),
    }
    for length in HISTORY_LENGTHS:
        fixtures[f"ring/L{length}"] = _history_fixture(
            length=length,
            with_context=False,
            name=f"ring/L{length}",
            role="history ladder",
        )
        fixtures[f"ring_witness/L{length}"] = _history_fixture(
            length=length,
            with_context=True,
            name=f"ring_witness/L{length}",
            role="identifiability witness",
        )
    return fixtures


def assert_row_stochastic(kernel: np.ndarray, *, name: str = "kernel", atol: float = 1e-9) -> None:
    if not np.allclose(np.asarray(kernel, dtype=float).sum(axis=1), 1.0, atol=atol):
        raise EstimandError(f"{name} rows must sum to 1 within {atol}")


def uniform_cell_weights(fixture: DiscreteFixture) -> np.ndarray:
    """Occupancy weights under the passive kernel."""
    weights = fixture.stationary_weights("passive")
    if weights.shape != (fixture.n_states,):
        raise EstimandError("stationary weights do not match the state count")
    return as_distribution(weights, name="stationary weights")
