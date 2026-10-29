"""Active support-set selection for the Phase 8 adaptive twin.

The candidate pool is always the animal's *support* pool, never its test events.
The ``eig`` strategy is an ensemble variance-reduction proxy: it prefers an
event whose predicted profile is most uncertain across independently trained
twins.  It is not a formal posterior expected-information-gain guarantee and is
labelled accordingly in the reports.

Candidate scoring is batched.  An early per-candidate implementation was
quadratic in the support pool and made the full run pathologically slow; the
batched version is both faster and easier to audit.
"""

from __future__ import annotations

import numpy as np

from neurobranch.twin.tasks import AnimalTask
from neurobranch.twin.train import TwinFit


def _mean_sigma_by_event(fit: TwinFit, events: list[str], support: list[str]) -> dict[str, float]:
    if not events:
        return {}
    _, frame = fit.predict_events(events, support_events=support)
    if frame.empty or "sigma" not in frame.columns:
        return {}
    return {
        str(event): float(value)
        for event, value in frame.groupby("event_id", sort=False)["sigma"].mean().items()
    }


def _mean_mu_by_event(fit: TwinFit, events: list[str], support: list[str]) -> dict[str, float]:
    if not events:
        return {}
    _, frame = fit.predict_events(events, support_events=support)
    if frame.empty or "y_pred" not in frame.columns:
        return {}
    return {
        str(event): float(value)
        for event, value in frame.groupby("event_id", sort=False)["y_pred"].mean().items()
    }


def _target_novelty(fit: TwinFit, candidate: str, acquired: set[str]) -> float:
    lookup = fit.data.event_lookup()
    i = lookup.get(str(candidate))
    if i is None:
        return 0.0
    target = fit.data.cache.target_names[int(fit.data.cache.target_idx[i])]
    seen_targets = set()
    for event in acquired:
        j = lookup.get(str(event))
        if j is not None:
            seen_targets.add(fit.data.cache.target_names[int(fit.data.cache.target_idx[j])])
    return 1.0 if target not in seen_targets else 0.0


def _score_candidates(
    fit: TwinFit,
    candidates: list[str],
    *,
    acquired: list[str],
    strategy: str,
    ensemble: list[TwinFit] | None,
) -> dict[str, float]:
    support = [str(e) for e in acquired]
    unc = _mean_sigma_by_event(fit, candidates, support)
    if strategy == "uncertainty":
        return {c: unc.get(str(c), 0.0) for c in candidates}
    if strategy == "diversity":
        return {c: _target_novelty(fit, c, {str(e) for e in acquired}) for c in candidates}
    if strategy == "eig":
        fits = ensemble or [fit]
        mus = [_mean_mu_by_event(f, candidates, support) for f in fits]
        scores = {}
        for c in candidates:
            values = np.array([m.get(str(c), np.nan) for m in mus], dtype=float)
            values = values[np.isfinite(values)]
            scores[c] = float(np.std(values)) if values.size > 1 else 0.0
        return scores
    if strategy == "hybrid":
        fits = ensemble or [fit]
        mus = [_mean_mu_by_event(f, candidates, support) for f in fits]
        u = np.array([unc.get(str(c), 0.0) for c in candidates], dtype=float)
        d = np.array(
            [_target_novelty(fit, c, {str(e) for e in acquired}) for c in candidates],
            dtype=float,
        )
        e = np.array(
            [
                float(
                    np.std(
                        [
                            m.get(str(c), np.nan)
                            for m in mus
                            if np.isfinite(m.get(str(c), np.nan))
                        ]
                    )
                )
                if len(mus) > 1
                else 0.0
                for c in candidates
            ]
        )

        def z(x: np.ndarray) -> np.ndarray:
            s = x.std()
            return (x - x.mean()) / (s if s > 0 else 1.0)

        combined = z(u) + z(d) + z(e)
        return {c: float(combined[i]) for i, c in enumerate(candidates)}
    raise ValueError(f"unknown support strategy {strategy!r}")


def select_support_event(
    fit: TwinFit,
    task: AnimalTask,
    *,
    acquired: list[str] | tuple[str, ...],
    strategy: str = "random",
    rng: np.random.Generator | None = None,
    ensemble: list[TwinFit] | None = None,
) -> str:
    """Select one support event from the target-disjoint support pool."""
    acquired_set = {str(e) for e in acquired}
    candidates = [str(e) for e in task.support_events if str(e) not in acquired_set]
    if not candidates:
        raise ValueError(f"no support candidates left for {task.animal}")
    if strategy == "random":
        local = rng or np.random.default_rng(17)
        return str(candidates[int(local.integers(len(candidates)))])
    scores = _score_candidates(
        fit,
        candidates,
        acquired=list(acquired),
        strategy=strategy,
        ensemble=ensemble,
    )
    ranked = sorted(candidates, key=lambda c: (-scores.get(c, 0.0), c))
    return str(ranked[0])


def select_support_prefix(
    fit: TwinFit,
    task: AnimalTask,
    *,
    n: int,
    strategy: str = "random",
    seed: int = 101,
    ensemble: list[TwinFit] | None = None,
) -> list[str]:
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for _ in range(min(n, len(task.support_events))):
        selected.append(
            select_support_event(
                fit, task, acquired=selected, strategy=strategy, rng=rng, ensemble=ensemble
            )
        )
    return selected
