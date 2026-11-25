"""Development tests for frozen semantics, information flow and guarded execution."""

import hashlib
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from neurobranch.abstraction.ic1_msc_oracle import (
    CONTRACTS,
    SUPPORT,
    information_cell,
    oracle_summary,
    response_law,
    tv,
)
from neurobranch.abstraction.ic1_v2_oracle import enumerate_response_laws
from scripts import ic1_msc_v2 as runner
from scripts.ic1_v2_acceptance import calibration_cases


def test_rederived_exact_risks_and_learning_objective_gap():
    summary = oracle_summary()
    assert [summary[arm]["bayes_optimum_exact"] for arm in CONTRACTS] == ["17/40", "17/40", "0", "7/20"]
    for scores in summary.values():
        assert scores["minimax_witness_risks_exact"] == [scores["bayes_optimum_exact"]] * 2
    assert summary["compressed_base"]["conditional_mean_risk_exact"] == "19/40"
    assert summary["compressed_base"]["objective_gap_exact"] == "1/20"
    assert all(summary[arm]["objective_gap_exact"] == "0" for arm in CONTRACTS if arm != "compressed_base")


def test_equal_passive_laws_pointwise_context_and_velocity_separation():
    histories = {}
    for initial_position, initial_velocity, context in product(range(3), (-1, 1), (0, 1)):
        for velocity, mass in ((initial_velocity, F(17, 20)), (-initial_velocity, F(3, 20))):
            cell = (initial_position, (initial_position + velocity) % 3)
            key = (context, cell)
            histories[key] = histories.get(key, F(0)) + mass / 6
    assert len(histories) == 12 and set(histories.values()) == {F(1, 6)}
    for position, velocity in product(range(3), (-1, 1)):
        assert tv(response_law(position, velocity, 0), response_law(position, velocity, 1)) == F(17, 20)
    for position, context in product(range(3), (0, 1)):
        assert tv(response_law(position, -1, context), response_law(position, 1, context)) == F(7, 10)


def test_rational_terminal_laws_agree_with_independent_kernel_oracle():
    ring = next(case for case in calibration_cases() if case.name == "ring_asym")
    historical = enumerate_response_laws(ring.design)
    for index, (position, velocity, context) in enumerate(product(range(3), (1, -1), (0, 1))):
        assert historical.response_by_start[index] == pytest.approx(
            tuple(float(value) for value in response_law(position, velocity, context))
        )


def test_estimator_fault_does_not_infect_exact_oracle(monkeypatch):
    original = oracle_summary()
    monkeypatch.setattr(runner, "predict", lambda table, rows: np.repeat([[1.0, 0.0, 0.0]], len(rows), axis=0))
    states = list(SUPPORT)
    exact = np.asarray([response_law(*state) for state in states], dtype=float)
    keys = [information_cell(state, "full_sensor") for state in states]
    scores = runner._score({}, keys, exact, "full_sensor", original["full_sensor"])
    assert scores["population_bayes_tv"] > 0.6
    assert oracle_summary() == original


def test_hidden_fields_cannot_change_features_or_learned_fit():
    data = runner.generate_episodes(np.random.default_rng(551), 32, training=True)
    assert data.outcome is not None
    altered_hidden = replace(data, velocity=-data.velocity, mechanism=1 - data.mechanism)
    for arm in CONTRACTS:
        assert runner.permitted_features(data, arm) == runner.permitted_features(altered_hidden, arm)
    altered_observation = replace(data, sensor=1 - data.sensor)
    for arm in ("full_base", "compressed_base"):
        base = runner.permitted_features(data, arm)
        changed = runner.permitted_features(altered_observation, arm)
        np.testing.assert_array_equal(
            list(runner.fit_response_table(base, data.outcome).values()),
            list(runner.fit_response_table(changed, data.outcome).values()),
        )
    for arm in ("full_sensor", "compressed_sensor"):
        assert runner.permitted_features(data, arm) != runner.permitted_features(altered_observation, arm)
    for arm in CONTRACTS:
        assert runner.permitted_features(data, arm) == [
            information_cell(state, arm) for state in runner._states(data)
        ]
    assert runner.generate_episodes(np.random.default_rng(552), 32, training=False).outcome is None


def test_exact_distribution_tv_is_not_one_hot_scoring():
    row = np.asarray([[0.85, 0.15, 0.0]])
    assert runner.exact_distribution_tv(row, row).tolist() == [0.0]
    assert runner.exact_distribution_tv(row, np.asarray([[0.0, 1.0, 0.0]])).tolist() == [0.85]


def test_independent_seed_key_design_without_initializing_heldout():
    seeds = runner._load(runner.SEEDS)
    plans = runner._load(runner.SPLITS)
    keys = {(namespace, n, role) for mode, namespace in seeds["namespaces"].items()
            for n in range(plans[mode]["blocks"]) for role in seeds["roles"].values()}
    assert len(keys) == 2 * (plans["dev"]["blocks"] + plans["heldout"]["blocks"])
    assert runner.namespace_rng(seeds, "dev", 0, "train").integers(2**32) != runner.namespace_rng(
        seeds, "dev", 0, "score"
    ).integers(2**32)


def test_unauthorized_branch_fails_before_verification_or_generation(monkeypatch, capsys):
    monkeypatch.delenv(runner.AUTHORIZATION_ENV, raising=False)

    def forbid(*args, **kwargs):
        pytest.fail("unauthorized code advanced past the gate")

    monkeypatch.setattr(runner, "verify_manifest", forbid)
    monkeypatch.setattr(runner, "generate_episodes", forbid)
    monkeypatch.setattr(runner, "namespace_rng", forbid)
    with pytest.raises(SystemExit) as error:
        runner.main(["--mode", "heldout"])
    assert error.value.code == 2
    assert "requires explicit external authorization" in capsys.readouterr().err
    with pytest.raises(PermissionError):
        runner._run("heldout", {}, {}, {})


def test_missing_or_tampered_manifest_entries_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "MANIFEST", Path("manifest.sha256"))
    monkeypatch.setattr(runner, "REQUIRED", {Path("fixture.txt")})
    original = b"frozen input"
    (tmp_path / "fixture.txt").write_bytes(original)
    manifest = tmp_path / runner.MANIFEST
    manifest.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks required"):
        runner.verify_manifest()
    manifest.write_text(f"{hashlib.sha256(original).hexdigest()}  fixture.txt\n", encoding="utf-8")
    runner.verify_manifest()
    (tmp_path / "fixture.txt").write_bytes(b"changed input")
    with pytest.raises(ValueError, match="digest mismatch"):
        runner.verify_manifest()


def test_canonical_hash_changes_on_splits_and_scientific_content():
    config, splits, seeds = (runner._load(path) for path in (runner.CONFIG, runner.SPLITS, runner.SEEDS))
    first = runner.scientific_fingerprint("alpha\nbeta\n", config, splits, seeds)
    second = runner.scientific_fingerprint("alpha\r\nbeta\r\n", dict(reversed(list(config.items()))), splits, seeds)
    assert first == second
    assert runner.scientific_fingerprint("alpha\nbeta  \n", config, splits, seeds) != first
    assert runner.scientific_fingerprint("alpha\nbeta\n", config, splits, {**seeds, "entropy": 3}) != first


def test_dev_reproducible_and_only_uses_dev_streams(monkeypatch):
    config, splits, seeds = (runner._load(path) for path in (runner.CONFIG, runner.SPLITS, runner.SEEDS))
    initialize = runner.namespace_rng
    seen = []

    def only_dev(seed_spec, mode, block, role):
        assert mode == "dev"
        seen.append((block, role))
        return initialize(seed_spec, mode, block, role)

    monkeypatch.setattr(runner, "namespace_rng", only_dev)
    first = runner._run("dev", config, splits, seeds)
    assert first == runner._run("dev", config, splits, seeds)
    assert len(set(seen)) == 8
    assert first["primary"]["independent_blocks"] == 4
    assert first["primary"]["heldout_decision"] is None
    assert not first["authorization_present"] and not first["realized_score_outcomes_generated"]
    assert runner._coverage(list(SUPPORT[:-1]))["omitted_passive_mass_exact"] == "1/12"
