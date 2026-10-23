from __future__ import annotations

import numpy as np

from neurobranch.branching import local_outcome_diversity, permutation_branching_test
from neurobranch.data.kato import MACRO_ORDER


def test_local_neighbourhood_returns_probabilities(toy_data):
    from sklearn.decomposition import PCA

    Z = PCA(3, random_state=0).fit_transform(toy_data.X)
    cand = np.flatnonzero(toy_data.y_macro == MACRO_ORDER.index("REV"))[:40]
    res = local_outcome_diversity(Z, toy_data, cand, k=8, horizon=10)
    assert res
    for r in res:
        assert abs(sum(r.probabilities.values()) - 1.0) < 1e-9
        assert 0 <= r.entropy <= np.log2(4) + 1e-9


def test_permutation_test_bounds(toy_data):
    from sklearn.decomposition import PCA

    Z = PCA(3, random_state=0).fit_transform(toy_data.X)
    cand = np.flatnonzero(toy_data.y_macro == MACRO_ORDER.index("REV"))[:30]
    out = permutation_branching_test(Z, toy_data, cand, k=8, horizon=10, n_permutations=20, seed=0)
    assert 0.0 <= out["p_value"] <= 1.0
    assert np.isfinite(out["mean_entropy"])
