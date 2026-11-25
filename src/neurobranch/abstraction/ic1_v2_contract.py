"""Inputs shared by the independent IC1-v2 oracle and trajectory estimator."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurobranch.abstraction.systems import DiscreteFixture


@dataclass(frozen=True)
class InterventionDesign:
    """Finite labelled checkpoint, operator sequence and terminal response law."""

    fixture: DiscreteFixture
    passive_weights: tuple[int, ...]
    operators: tuple[str, ...]
    emission: NDArray[np.float64]

    def __post_init__(self) -> None:
        weights = self.passive_weights
        if len(weights) != self.fixture.n_states or any(type(weight) is not int or weight < 0 for weight in weights):
            raise ValueError("passive weights must be nonnegative integers, one per microstate")
        if not 0 < sum(weights) <= np.iinfo(np.int64).max:
            raise ValueError("passive weight total must be positive and fit in signed int64")

        partition = np.asarray(self.fixture.partition)
        if not np.issubdtype(partition.dtype, np.integer):
            raise ValueError("cell labels must be integers")
        if not self.operators or any(operator not in self.fixture.operators for operator in self.operators):
            raise ValueError("provide a nonempty sequence of declared transition operators")
        for operator in dict.fromkeys(self.operators):
            kernel = self.fixture.kernel(operator)
            if (kernel < 0).any() or not np.allclose(kernel.sum(axis=1), 1, atol=1e-12, rtol=0):
                raise ValueError(f"transition operator {operator!r} must have nonnegative stochastic rows")

        emission = np.array(self.emission, dtype=float, copy=True)
        if emission.ndim != 2 or emission.shape[0] != self.fixture.n_states or emission.shape[1] == 0:
            raise ValueError("provide one emission row per microstate on a nonempty response alphabet")
        totals = emission.sum(axis=1)
        if not np.isfinite(emission).all() or (emission < 0).any() or not np.allclose(totals, 1, atol=1e-12, rtol=0):
            raise ValueError("terminal emission rows must be probability distributions")
        emission /= totals[:, None]
        emission.setflags(write=False)
        object.__setattr__(self, "emission", emission)


def protocol_fingerprint(protocol_utf8: bytes) -> str:
    """Hash strict UTF-8 text with platform-independent newlines."""
    text = protocol_utf8.decode("utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
