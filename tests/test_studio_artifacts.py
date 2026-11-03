"""Contract tests for the Scientific Studio evidence package."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from neurobranch.studio import load_snapshot, load_trajectory_store
from neurobranch.studio.metadata import ArtifactUnavailable

pytestmark = pytest.mark.requires_data


def test_dataset_identity_hash() -> None:
    snapshot = load_snapshot()
    archive = snapshot.paths.data
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == snapshot.dataset_identity["archive"]["actual_sha256"]
    assert archive.stat().st_size == 56_390_975
    assert snapshot.dataset_ok


def test_primary_contrast_is_recomputed_not_typed() -> None:
    snapshot = load_snapshot()
    primary = snapshot.primary
    assert primary["n"] == 175
    assert primary["b3_mean"] == pytest.approx(2.700149750441149, abs=1e-12)
    assert primary["b5_mean"] == pytest.approx(2.9645877431095045, abs=1e-12)
    assert primary["difference"] == pytest.approx(0.2644379926683563, abs=1e-12)
    assert primary["relative_to_b3"] == pytest.approx(0.09793456552739439, abs=1e-12)
    assert primary["sessions_positive"] == 4
    assert primary["conditional_ci"]["ci_low"] == pytest.approx(0.07499291957903519, abs=1e-12)
    assert primary["recording_ci"]["ci_low"] == pytest.approx(-0.040800908320236015, abs=1e-12)


def test_landmark_counts_and_session_identity() -> None:
    store = load_trajectory_store()
    assert [len(store.session(sid)["landmarks"]) for sid in range(5)] == [35, 35, 34, 38, 33]
    assert [store.session(sid)["n_frames"] for sid in range(5)] == [3137, 3134, 3059, 3311, 3021]
    assert sum(store.session(sid)["n_predictors"] for sid in range(5)) == 592
    assert sum(store.session(sid)["n_frames"] for sid in range(5)) == 15_662


def test_trajectory_horizon_contract() -> None:
    store = load_trajectory_store()
    assert store.metadata["rollout_horizon"] == 100
    assert store.metadata["evaluation_horizon"] == 20
    case = store.case(0, 2690)
    assert case["truth"].shape == (20, 10)
    assert all(value.shape == (20, 10) for value in case["means"].values())
    showcase_case = store.case(0, store.session(0)["showcase_landmarks"][0])
    for model, ensemble in showcase_case["ensembles"].items():
        assert ensemble.shape == (100, 20, 10), model
    assert case["explained_3d"] == pytest.approx(
        float(np.sum(case["explained_variance_ratio"][:3])), abs=1e-6
    )


def test_observed_and_forecast_are_aligned() -> None:
    store = load_trajectory_store()
    sid, t = 3, store.session(3)["landmarks"][10]
    case = store.case(sid, t)
    observed = store.observed(sid)
    np.testing.assert_allclose(case["truth"], observed[t + 1 : t + 21], atol=1e-6)
    assert case["origin_time"] == pytest.approx(store.time_seconds(sid)[t])
    assert case["future_time"][0] > case["origin_time"]
    assert np.all(np.diff(case["context_time"]) > 0)


def test_every_landmark_has_finite_model_scores() -> None:
    snapshot = load_snapshot()
    for model in ("B0_persistence", "B1_matched_own", "B3_population", "B5_common_k10"):
        values = snapshot.per_landmark.loc[
            snapshot.per_landmark["model"] == model, "pca10_energy_h20"
        ]
        assert len(values) == 175
        assert np.isfinite(values).all(), model


def test_storage_validation_was_recorded() -> None:
    store = load_trajectory_store()
    validation = store.metadata["score_validation"]
    assert validation["n_showcase_checks"] == 30
    assert validation["max_abs_difference"] <= validation["tolerance"]
    assert validation["tolerance"] == pytest.approx(2e-5)


def test_missing_artifact_is_explicit(tmp_path) -> None:
    with pytest.raises(ArtifactUnavailable, match="Required artifact is missing"):
        load_snapshot(tmp_path)
