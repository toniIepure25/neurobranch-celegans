"""Development and authorization-gated runner for the finite IC1 MSc study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.stats import t

from neurobranch.abstraction.ic1_msc_oracle import (
    CONTRACTS,
    SUPPORT,
    information_cell,
    oracle_summary,
    response_law,
)

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = Path("research/ic1_v2")
PROTOCOL = RESEARCH / "MINIMAL_DECISIVE_EXPERIMENT_PROTOCOL_V2_FROZEN.md"
CONFIG = RESEARCH / "protocol_v2_frozen.json"
SPLITS = RESEARCH / "splits_v2.json"
SEEDS = RESEARCH / "seeds_v2.json"
MANIFEST = RESEARCH / "PROTOCOL_V2_FREEZE_MANIFEST.sha256"
HELDOUT_OUTPUT = RESEARCH / "HELDOUT_RESULTS_V2.json"
AUTHORIZATION_ENV = "NEUROBRANCH_IC1_V2_HELDOUT_AUTHORIZATION"
REQUIRED = {PROTOCOL, CONFIG, SPLITS, SEEDS, Path("scripts/ic1_msc_v2.py"),
            Path("src/neurobranch/abstraction/ic1_msc_oracle.py"),
            Path("src/neurobranch/__init__.py"), Path("src/neurobranch/abstraction/__init__.py")}
ALPHABET = 3
PSEUDOCOUNT = 0.5


@dataclass(frozen=True)
class Episodes:
    previous: NDArray[np.int64]
    current: NDArray[np.int64]
    velocity: NDArray[np.int64]
    mechanism: NDArray[np.int64]
    sensor: NDArray[np.int64]
    outcome: NDArray[np.int64] | None


def _authorized() -> bool:
    return bool(os.environ.get(AUTHORIZATION_ENV, "").strip())


def verify_manifest() -> None:
    entries = {}
    for line in (ROOT / MANIFEST).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path in entries:
            raise ValueError("unsafe or duplicate manifest entry")
        resolved = (ROOT / path).resolve()
        if not resolved.is_relative_to(ROOT) or hashlib.sha256(resolved.read_bytes()).hexdigest() != digest:
            raise ValueError(f"freeze digest mismatch for {name}")
        entries[path] = digest
    if REQUIRED - entries.keys():
        raise ValueError("freeze manifest lacks required scientific inputs")


def _load(path: Path) -> dict:
    def unique_keys(pairs: list[tuple[str, object]]) -> dict:
        fields = dict(pairs)
        if len(fields) != len(pairs):
            raise ValueError(f"ambiguous duplicated key in {path}")
        return fields

    return json.loads((ROOT / path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def scientific_fingerprint(protocol: str, config: dict, splits: dict, seeds: dict) -> str:
    canonical_text = protocol.replace("\r\n", "\n").replace("\r", "\n")
    canonical = json.dumps([canonical_text, config, splits, seeds], sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_config(config: dict, splits: dict, seeds: dict) -> None:
    model = config["model"]
    if (model["positions"], model["velocities"], model["contexts"], model["context_prior"],
        model["velocity_persistence"], model["passive_steps"], model["intervention_steps"]) != (
        3, [-1, 1], [0, 1], ["1/2", "1/2"], "17/20", 1, 1
    ):
        raise ValueError("probability model differs from the exact finite generator")
    if tuple(config["learner"]["inputs"]) != CONTRACTS or config["learner"]["pseudocount_per_category"] != PSEUDOCOUNT:
        raise ValueError("learner contract differs from the frozen table")
    analysis = config["analysis"]
    if analysis["heldout_realized_outcomes"] is not False or analysis["confidence_level"] != 0.95:
        raise ValueError("analysis contract requires exact truth and the fixed interval level")
    if seeds["key_order"] != ["namespace", "zero_based_block", "role"] or seeds["roles"] != {"train": 0, "score": 1}:
        raise ValueError("seed configuration has an unsupported key layout")
    namespaces = seeds["namespaces"]
    if set(namespaces) != {"dev", "heldout"} or len(set(namespaces.values())) != 2:
        raise ValueError("development and held-out namespaces must be disjoint")
    for mode in namespaces:
        plan = splits[mode]
        sizes = plan["train_sizes"]
        if sorted(set(sizes)) != sizes or not sizes or min(sizes) < 1 or min(
            plan["blocks"], plan["score_episodes_per_block"]
        ) < 2:
            raise ValueError(f"invalid split sizes for {mode}")


def namespace_rng(seeds: dict, mode: str, block: int, role: str) -> np.random.Generator:
    key = (seeds["namespaces"][mode], block, seeds["roles"][role])
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seeds["entropy"], spawn_key=key)))


def generate_episodes(rng: np.random.Generator, count: int, *, training: bool) -> Episodes:
    """Generate fresh passive checkpoints and, for training only, probe labels."""
    previous = rng.integers(3, size=count)
    start_velocity = 2 * rng.integers(2, size=count) - 1
    mechanism = rng.integers(2, size=count)
    velocity = np.where(rng.integers(20, size=count) < 17, start_velocity, -start_velocity)
    current = (previous + velocity) % 3
    outcome = None
    if training:
        step_velocity = np.where(rng.integers(20, size=count) < 17, velocity, -velocity)
        outcome = (current + step_velocity + mechanism) % 3
    return Episodes(previous, current, velocity, mechanism, mechanism.copy(), outcome)


def permitted_features(episodes: Episodes, contract: str) -> list[tuple[int, ...]]:
    """Project only the agreed observable fields, without identifiers or metadata."""
    if contract not in CONTRACTS:
        raise ValueError("unknown information contract")
    fields = [episodes.current]
    if contract.startswith("full_"):
        fields.insert(0, episodes.previous)
    if contract.endswith("_sensor"):
        fields.append(episodes.sensor)
    return list(zip(*(field.tolist() for field in fields), strict=True))


def fit_response_table(features: list[tuple[int, ...]], outcomes: NDArray[np.int64]) -> dict:
    """Fit the single-response categorical table from permitted training rows."""
    counts = {}
    for cell, label in zip(features, outcomes, strict=True):
        counts.setdefault(cell, np.full(ALPHABET, PSEUDOCOUNT))[label] += 1
    return {cell: row / row.sum() for cell, row in counts.items()}


def predict(table: dict, features: list[tuple[int, ...]]) -> NDArray[np.float64]:
    fallback = np.full(ALPHABET, 1 / ALPHABET)
    return np.asarray([table.get(cell, fallback) for cell in features])


def exact_distribution_tv(predictions: NDArray[np.float64], truth: NDArray[np.float64]) -> NDArray[np.float64]:
    if predictions.shape != truth.shape or predictions.ndim != 2 or predictions.shape[1] != ALPHABET:
        raise ValueError("expected paired categorical distributions with three outcomes")
    return np.abs(predictions - truth).sum(axis=1) / 2


def _states(episodes: Episodes) -> list[tuple[int, int, int]]:
    return list(zip(episodes.current.tolist(), episodes.velocity.tolist(), episodes.mechanism.tolist(), strict=True))


def _coverage(states: list[tuple[int, int, int]]) -> dict:
    observed = set(states)
    missing = sorted(set(SUPPORT) - observed)
    mass = Fraction(len(missing), len(SUPPORT))
    return {"seen": sorted(observed), "missed": missing, "omitted_passive_mass_exact": str(mass),
            "omitted_pair_contribution_upper": float(min(1, 2 * mass)),
            "bound_exceeds_0.05_tv": 2 * mass > Fraction(1, 20)}


def _score(table: dict, evaluation_keys: list[tuple[int, ...]], exact_truth: NDArray[np.float64],
           contract: str, summary: dict) -> dict:
    distribution_tv = exact_distribution_tv(predict(table, evaluation_keys), exact_truth)
    support_keys = [information_cell(state, contract) for state in SUPPORT]
    support_truth = np.asarray([response_law(*state) for state in SUPPORT], dtype=float)
    population_tv = exact_distribution_tv(predict(table, support_keys), support_truth)
    by_mechanism = [
        float(population_tv[[state[2] == context for state in SUPPORT]].mean())
        for context in (0, 1)
    ]
    bayes = float(population_tv.mean())
    optimal = float(Fraction(summary["bayes_optimum_exact"]))
    mixture = float(Fraction(summary["conditional_mean_risk_exact"]))
    return {"episode_mean_tv": float(distribution_tv.mean()), "population_bayes_tv": bayes,
            "population_tv_by_mechanism": by_mechanism, "worst_fixed_mechanism_tv": max(by_mechanism),
            "excess_over_own_cell_oracle": bayes - optimal,
            "finite_fit_minus_population_mixture": bayes - mixture,
            "objective_mixture_minus_cell_oracle": mixture - optimal,
            "unobserved_training_cells": sorted(set(support_keys) - table.keys())}


def _paired_t(values: list[float], level: float) -> dict:
    n = len(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    delta = float(t.ppf((1 + level) / 2, n - 1) * se)
    return {"estimate": mean, "confidence_level": level, "lower": mean - delta, "upper": mean + delta,
            "independent_blocks": n, "type": "two-sided paired-block Student t"}


def _run(mode: str, config: dict, splits: dict, seeds: dict) -> dict:
    if mode == "heldout" and not _authorized():
        raise PermissionError("held-out execution requires explicit external authorization")
    plan = splits[mode]
    deadline = time.monotonic() + config["runtime"]["budget_seconds"]
    oracles = oracle_summary()
    expected = {"full_base": "17/40", "compressed_base": "17/40", "full_sensor": "0",
                "compressed_sensor": "7/20"}
    if any(oracles[arm]["bayes_optimum_exact"] != value for arm, value in expected.items()):
        raise ValueError("exact benchmark risks differ from the frozen references")

    blocks = []
    for block in range(plan["blocks"]):
        if time.monotonic() > deadline:
            raise TimeoutError("local CPU budget exhausted")
        training = generate_episodes(namespace_rng(seeds, mode, block, "train"), max(plan["train_sizes"]),
                                     training=True)
        evaluation = generate_episodes(namespace_rng(seeds, mode, block, "score"),
                                       plan["score_episodes_per_block"], training=False)
        assert training.outcome is not None and evaluation.outcome is None
        states = _states(evaluation)
        exact_truth = np.asarray([response_law(*state) for state in states], dtype=float)
        train_features = {arm: permitted_features(training, arm) for arm in CONTRACTS}
        test_features = {arm: permitted_features(evaluation, arm) for arm in CONTRACTS}
        train_states = _states(training)
        sizes = {}
        for size in plan["train_sizes"]:
            scores = {}
            for arm in CONTRACTS:
                table = fit_response_table(train_features[arm][:size], training.outcome[:size])
                scores[arm] = _score(table, test_features[arm], exact_truth, arm, oracles[arm])
                full = f"full_{arm.rsplit('_', 1)[-1]}"
                own_optimum = Fraction(oracles[arm]["bayes_optimum_exact"])
                full_optimum = Fraction(oracles[full]["bayes_optimum_exact"])
                scores[arm]["compression_gap_exact"] = str(own_optimum - full_optimum)
                scores[arm]["model_class_approximation_gap_exact"] = "0"
                scores[arm]["total_learned_excess_over_full_oracle"] = (
                    scores[arm]["population_bayes_tv"] - float(full_optimum)
                )
            sizes[str(size)] = {"scores": scores, "training_coverage": _coverage(train_states[:size])}
        blocks.append({"block": block, "size_results": sizes, "evaluation_coverage": _coverage(states)})

    def primary_difference(block: dict) -> float:
        scores = block["size_results"][str(max(plan["train_sizes"]))]["scores"]
        return scores["compressed_sensor"]["episode_mean_tv"] - scores["full_sensor"]["episode_mean_tv"]

    primary = _paired_t([primary_difference(block) for block in blocks], config["analysis"]["confidence_level"])
    reference = Fraction(config["analysis"]["oracle_compression_reference"])
    margin = Fraction(config["analysis"]["agreement_margin"])
    lower_band, upper_band = float(reference - margin), float(reference + margin)
    if primary["lower"] >= lower_band and primary["upper"] <= upper_band:
        outcome = "interval_within_prespecified_agreement_band"
    elif primary["upper"] < lower_band:
        outcome = "interval_below_prespecified_agreement_band"
    elif primary["lower"] > upper_band:
        outcome = "interval_above_prespecified_agreement_band"
    else:
        outcome = "inconclusive_agreement"
    primary.update({"training_episodes": max(plan["train_sizes"]), "agreement_band": [lower_band, upper_band],
                    "heldout_decision": outcome if mode == "heldout" else None})
    return {"mode": mode, "independent_blocks": blocks, "oracles": oracles, "primary": primary,
            "authorization_present": mode == "heldout", "realized_score_outcomes_generated": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--validate", action="store_true", help="verify the frozen inputs without generating episodes")
    parser.add_argument("--output", type=Path, help="new development output path")
    args = parser.parse_args(argv)
    if args.mode == "heldout":
        if not _authorized():
            parser.error("held-out execution requires explicit external authorization")
        if args.output is not None:
            parser.error("held-out output path is fixed by the protocol")

    verify_manifest()
    config, splits, seeds = _load(CONFIG), _load(SPLITS), _load(SEEDS)
    validate_config(config, splits, seeds)
    runtime = {"python": platform.python_version(), "numpy": np.__version__,
               "scipy": scipy.__version__, "system": platform.system()}
    if any(value != config["runtime"][key] for key, value in runtime.items()):
        parser.error("runtime differs from the frozen environment")
    fingerprint = scientific_fingerprint((ROOT / PROTOCOL).read_text(encoding="utf-8"), config, splits, seeds)
    if args.validate:
        print(f"Frozen inputs verified: {fingerprint}")
        return 0

    def execute(handle) -> None:
        result = _run(args.mode, config, splits, seeds)
        result.update({"scientific_fingerprint": fingerprint, "runtime": runtime})
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")

    destination = ROOT / HELDOUT_OUTPUT if args.mode == "heldout" else args.output
    if destination is not None:
        if args.mode == "dev" and destination.resolve() == (ROOT / HELDOUT_OUTPUT).resolve():
            parser.error("development cannot write to the held-out result path")
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            execute(handle)
    else:
        execute(sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
