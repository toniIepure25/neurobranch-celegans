"""Assumption audit for the NC-MCM macroscopic-state model.

NC-MCM turns behaviour probabilities into clusters, treats the cluster sequence
as a Markov chain, and reads behaviour from those states.  Those are *modelling
assumptions*, not established facts.  This module tests the assumptions that a
downstream counterfactual depends on:

* first-order Markov structure;
* stationarity across a recording;
* transport of behaviour statistics to a held-out worm;
* state-behaviour information;
* sensitivity to the chosen number of clusters ``K``.

The output is an audit, not a causal identification proof.  A ``fail`` means the
corresponding model-based intervention should be read as descriptive only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .dataset import BEHAVIOR_NAMES
from .ncmcm import NCMCMWorm, load_ncmcm

try:  # sklearn is already a declared dependency for the Tier-1 models
    from sklearn.metrics import mutual_info_score
except ImportError:  # pragma: no cover - dependency is pinned by the project
    mutual_info_score = None


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))))

    return 0.5 * (kl(p, m) + kl(q, m))


def _counts(seq: np.ndarray, k: int) -> np.ndarray:
    counts = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(seq[:-1], seq[1:]):
        counts[int(a) - 1, int(b) - 1] += 1
    return counts


def _loglik_order1(counts: np.ndarray) -> float:
    total = counts.sum(axis=1)
    value = 0.0
    for i in range(counts.shape[0]):
        if total[i] == 0:
            continue
        for j in range(counts.shape[1]):
            if counts[i, j]:
                value += counts[i, j] * np.log(counts[i, j] / total[i])
    return float(value)


def _loglik_order2(seq: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    counts2 = np.zeros((k, k, k), dtype=np.int64)
    for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
        counts2[int(a) - 1, int(b) - 1, int(c) - 1] += 1
    value = 0.0
    for i in range(k):
        for j in range(k):
            total = counts2[i, j].sum()
            if total == 0:
                continue
            for c in range(k):
                if counts2[i, j, c]:
                    value += counts2[i, j, c] * np.log(counts2[i, j, c] / total)
    return float(value), counts2


def markov_order_test(
    worm: NCMCMWorm,
    k: int,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, object]:
    """Test whether a second-order chain explains substantially more.

    The asymptotic chi-square p-value is reported, plus a Monte-Carlo p-value
    that conditions on the observed first-order counts.  The latter is the
    primary value because the cluster series is short and sparse.
    """
    seq = worm.cluster_sequence(k)
    if len(seq) < 20:
        return {"verdict": "insufficient_data", "reason": "fewer than 20 frames"}
    counts1 = _counts(seq, k)
    ll1 = _loglik_order1(counts1)
    ll2, counts2 = _loglik_order2(seq, k)
    lr = max(0.0, 2.0 * (ll2 - ll1))
    df = k * (k - 1) * (k - 1)
    p_chi2 = float(chi2.sf(lr, df)) if df > 0 else float("nan")

    row_totals = counts1.sum(axis=1)
    sparse_fraction = float(np.mean(row_totals < 5)) if len(row_totals) else 1.0
    if sparse_fraction > 0.50:
        return {
            "verdict": "insufficient_data",
            "lr": float(lr),
            "df": int(df),
            "p_chi2": p_chi2,
            "sparse_fraction": sparse_fraction,
            "reason": "more than half of states have fewer than 5 outgoing transitions",
        }

    rng = np.random.default_rng(seed)
    P = counts1.astype(float) + 0.5
    P /= P.sum(axis=1, keepdims=True)
    null_lr = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sim = np.empty_like(seq)
        sim[0] = seq[0]
        for t in range(1, len(sim)):
            sim[t] = rng.choice(k, p=P[int(sim[t - 1]) - 1]) + 1
        c1 = _counts(sim, k)
        l1 = _loglik_order1(c1)
        l2, _ = _loglik_order2(sim, k)
        null_lr[b] = max(0.0, 2.0 * (l2 - l1))
    p_mc = float((1.0 + np.sum(null_lr >= lr)) / (n_boot + 1.0))
    verdict = "fail" if p_mc < 0.05 else "pass"
    return {
        "verdict": verdict,
        "lr": float(lr),
        "df": int(df),
        "p_chi2": p_chi2,
        "p_monte_carlo": p_mc,
        "sparse_fraction": sparse_fraction,
        "n_boot": int(n_boot),
    }


def _behavior_transition(worm: NCMCMWorm, start: int = 0, stop: int | None = None) -> np.ndarray:
    labels = worm.behavior[start:stop]
    counts = np.full((len(BEHAVIOR_NAMES), len(BEHAVIOR_NAMES)), 1e-9)
    for a, b in zip(labels[:-1], labels[1:]):
        counts[int(a), int(b)] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def behavior_transport_test(
    worms: list[NCMCMWorm],
    *,
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, object]:
    """Held-out-worm transport of aligned behaviour transition statistics.

    Cluster labels are not aligned across worms, so transporting a state index
    is not meaningful.  Behaviour labels *are* aligned.  We therefore compare
    the held-out worm's behaviour transition matrix with the pooled matrix from
    the other worms, and calibrate the divergence against a split-half baseline
    inside the same worm.
    """
    if len(worms) < 3:
        return {"verdict": "insufficient_data", "reason": "need at least 3 worms"}
    rng = np.random.default_rng(seed)
    heldout_js: list[float] = []
    baseline_js: list[float] = []
    for held in range(len(worms)):
        train = [w for i, w in enumerate(worms) if i != held]
        pooled = np.full_like(_behavior_transition(worms[held]), 1e-9)
        for worm in train:
            pooled += _behavior_transition(worm)
        pooled /= pooled.sum(axis=1, keepdims=True)
        observed = _js(pooled.reshape(-1), _behavior_transition(worms[held]).reshape(-1))
        heldout_js.append(observed)

        labels = worms[held].behavior
        half = len(labels) // 2
        first = _behavior_transition(worms[held], 0, half)
        second = _behavior_transition(worms[held], half, None)
        baseline_js.append(_js(first.reshape(-1), second.reshape(-1)))
    heldout = float(np.mean(heldout_js))
    baseline = float(np.mean(baseline_js))
    # A held-out worm is transportable when it is no worse than the within-worm
    # split-half divergence (with a small tolerance for finite-sample noise).
    verdict = "pass" if heldout <= max(1.5 * baseline, baseline + 0.05) else "warn"
    return {
        "verdict": verdict,
        "heldout_js": heldout,
        "split_half_js": baseline,
        "per_worm_heldout_js": [float(v) for v in heldout_js],
        "interpretation": (
            "behaviour-transition transport only; state indices are not aligned across worms"
        ),
    }


def stationarity_test(
    worm: NCMCMWorm,
    k: int,
    *,
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, object]:
    """Compare transition matrices in the first and second half of a recording."""
    seq = worm.cluster_sequence(k)
    if len(seq) < 40:
        return {"verdict": "insufficient_data", "reason": "fewer than 40 frames"}
    half = len(seq) // 2
    first = _counts(seq[:half], k).astype(float) + 1e-9
    second = _counts(seq[half:], k).astype(float) + 1e-9
    first /= first.sum(axis=1, keepdims=True)
    second /= second.sum(axis=1, keepdims=True)
    observed = _js(first.reshape(-1), second.reshape(-1))

    rng = np.random.default_rng(seed)
    block = max(10, len(seq) // 20)
    null = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, len(seq) - block, size=len(seq) // block)
        pieces = [seq[s : s + block] for s in starts]
        sim = np.concatenate(pieces)[: len(seq)]
        h = len(sim) // 2
        a = _counts(sim[:h], k).astype(float) + 1e-9
        c = _counts(sim[h:], k).astype(float) + 1e-9
        a /= a.sum(axis=1, keepdims=True)
        c /= c.sum(axis=1, keepdims=True)
        null[b] = _js(a.reshape(-1), c.reshape(-1))
    p = float((1.0 + np.sum(null >= observed)) / (n_boot + 1.0))
    if p < 0.05 and observed > 0.05:
        verdict = "fail"
    elif p < 0.10:
        verdict = "warn"
    else:
        verdict = "pass"
    return {
        "verdict": verdict,
        "js_first_second_half": float(observed),
        "p_monte_carlo": p,
        "n_boot": int(n_boot),
    }


def state_behavior_information(
    worm: NCMCMWorm,
    k: int,
    *,
    n_perm: int = 500,
    seed: int = 0,
) -> dict[str, object]:
    if mutual_info_score is None:
        return {"verdict": "insufficient_data", "reason": "sklearn unavailable"}
    states = worm.cluster_sequence(k)
    behavior = worm.behavior
    observed = float(mutual_info_score(states, behavior))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = mutual_info_score(states, rng.permutation(behavior))
    p = float((1.0 + np.sum(null >= observed)) / (n_perm + 1.0))
    verdict = "pass" if p < 0.05 and observed > 0.01 else "warn"
    return {
        "verdict": verdict,
        "mutual_information": observed,
        "p_permutation": p,
        "n_perm": int(n_perm),
    }


def k_sensitivity(
    worms: list[NCMCMWorm],
    *,
    reference_k: int = 6,
    horizon: int = 20,
) -> dict[str, object]:
    """Maximum TV change in 20-step behaviour distributions across K."""
    if not worms:
        return {"verdict": "insufficient_data", "reason": "no worms"}
    reference = []
    for worm in worms:
        reference.append(
            np.mean(
                [
                    worm.intervention_horizon(reference_k, s, horizon)["behavior_distribution"][-1]
                    for s in range(1, reference_k + 1)
                ],
                axis=0,
            )
        )
    reference_mean = np.mean(reference, axis=0)
    changes: list[float] = []
    for k in range(2, 21):
        if k == reference_k:
            continue
        current = np.mean(
            [
                np.mean(
                    [
                        worm.intervention_horizon(k, s, horizon)["behavior_distribution"][-1]
                        for s in range(1, k + 1)
                    ],
                    axis=0,
                )
                for worm in worms
            ],
            axis=0,
        )
        changes.append(float(0.5 * np.abs(current - reference_mean).sum()))
    max_tv = max(changes) if changes else 0.0
    verdict = "pass" if max_tv <= 0.20 else "warn" if max_tv <= 0.35 else "fail"
    return {
        "verdict": verdict,
        "max_tv_vs_reference": max_tv,
        "reference_k": int(reference_k),
        "k_values": list(range(2, 21)),
        "change_by_k": [float(v) for v in changes],
    }


def audit_model(worm: NCMCMWorm, k: int = 6) -> dict[str, object]:
    return {
        "worm_id": int(worm.worm_id),
        "k": int(k),
        "n_frames": int(len(worm.behavior)),
        "markov_order": markov_order_test(worm, k),
        "stationarity": stationarity_test(worm, k),
        "state_behavior_mi": state_behavior_information(worm, k),
    }


def audit_all(
    worms: list[NCMCMWorm],
    *,
    k: int = 6,
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, object]:
    per_worm = [audit_model(w, k) for w in worms]
    tests = {
        "markov_order": _aggregate_verdicts([w["markov_order"] for w in per_worm]),
        "stationarity": _aggregate_verdicts([w["stationarity"] for w in per_worm]),
        "behavior_transport": behavior_transport_test(worms, n_boot=n_boot, seed=seed),
        "state_behavior_mi": _aggregate_verdicts([w["state_behavior_mi"] for w in per_worm]),
        "k_sensitivity": k_sensitivity(worms, reference_k=k),
    }
    return {
        "format": "vienna-tier2-assumption-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "k": int(k),
        "n_worms": len(worms),
        "per_worm": per_worm,
        "tests": tests,
        "caveats": [
            "Observational audit; not a causal identification proof.",
            "Cluster indices are not semantically aligned across worms.",
            "State-level counterfactuals remain model-based even when tests pass.",
            "Five worms limit population-level transport claims.",
        ],
    }


def _aggregate_verdicts(results: list[dict[str, object]]) -> dict[str, object]:
    verdicts = [str(r.get("verdict", "insufficient_data")) for r in results]
    counts = {v: verdicts.count(v) for v in ("pass", "warn", "fail", "insufficient_data")}
    if counts["fail"]:
        verdict = "fail"
    elif counts["warn"]:
        verdict = "warn"
    elif counts["pass"]:
        verdict = "pass"
    else:
        verdict = "insufficient_data"
    return {"verdict": verdict, "counts": counts, "per_worm": results}


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/vienna_workbench/tier2_assumption_audit.json"),
    )
    args = parser.parse_args()
    worms = [load_ncmcm(i) for i in range(5)]
    payload = audit_all(worms, k=args.k, n_boot=args.n_boot, seed=args.seed)
    _write_json(payload, args.out)
    print(json.dumps({k: v["verdict"] for k, v in payload["tests"].items()}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
