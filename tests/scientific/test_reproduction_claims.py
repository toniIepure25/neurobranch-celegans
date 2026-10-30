from __future__ import annotations

import numpy as np
import pytest

from neurobranch.config import PROJECT_ROOT

KATO_PATH = PROJECT_ROOT / "data" / "raw" / "kato2015" / "WT_NoStim.mat"
requires_kato = pytest.mark.skipif(not KATO_PATH.exists(), reason="Kato 2015 dataset not present")


@requires_kato
def test_behavior_label_neurons_are_excluded_from_real_predictors():
    from neurobranch.data.kato import BEHAVIOR_LABEL_NEURONS, load_kato
    from neurobranch.preprocessing import make_worm_data

    rec = load_kato()[0]
    data = make_worm_data(rec)
    assert not (set(data.predictor_names) & set(BEHAVIOR_LABEL_NEURONS))


@requires_kato
def test_real_data_split_windows_are_clean():
    from neurobranch.data.kato import load_kato
    from neurobranch.preprocessing import make_worm_data, split_starts

    data = make_worm_data(load_kato()[0])
    for split in ("train", "val", "test"):
        for t in split_starts(data, split, history=3, horizon=2)[:50]:
            assert data.masks[split][t - 2 : t + 2].all()


def test_ncmcm_markov_port_agrees_with_distributed_artifact():
    from neurobranch.reproduction.ncmcm import summarize_worm

    p = PROJECT_ROOT / "references" / "external" / "nc-mcm" / "precomputed_results" / "nc_mcm_model_worm_1.npz"
    if not p.exists():
        import pytest

        pytest.skip("precomputed NC-MCM artifact absent")
    s = summarize_worm(1, selected_k=3, n_markov_sims=100)
    assert abs(s["markov_p_precomputed"] - s["markov_p_reimplemented"]) < 0.12


@requires_kato
def test_random_controls_exclude_hypothesis_candidates():
    from neurobranch.data.kato import load_kato
    from neurobranch.interventions.search import available_candidates, random_control_pool
    from neurobranch.preprocessing import make_worm_data

    data = make_worm_data(load_kato()[0])
    pool = random_control_pool(data)
    assert pool, "worm 0 should have non-candidate named cells"
    assert not (set(pool) & set(available_candidates(data)))


@requires_kato
def test_official_bundlenet_evaluation_is_labelled_leaky():
    from neurobranch.representation.bundlenet import official_vs_reimplementation

    r = official_vs_reimplementation(0)
    if not r.get("available"):
        import pytest

        pytest.skip("official BunDLe artifact absent")
    assert "leakage" in r["evaluation_note"].lower()
    assert "not held-out" in r["evaluation_note"].lower()


def test_bundlenet_filter_is_causal():
    from neurobranch.representation.bundlenet import causal_lowpass

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    base = causal_lowpass(X, fs=3.0)
    changed = X.copy()
    changed[100:] += 50.0
    after = causal_lowpass(changed, fs=3.0)
    # A future change must not alter any past filtered sample.
    assert np.allclose(base[:100], after[:100])
    assert not np.allclose(base[100:], after[100:])


def test_official_bundlenet_artifact_shape_matches_frames():
    from neurobranch.data.kato import load_kato
    from neurobranch.representation.bundlenet import load_official_embedding

    got = load_official_embedding(0)
    if got is None:
        import pytest

        pytest.skip("official BunDLe artifact absent")
    Z, b, _ = got
    rec = load_kato()[0]
    assert Z.shape == (rec.n_frames - 16, 3)
    assert b.shape == (rec.n_frames - 16,)
    assert np.isfinite(Z).all()
