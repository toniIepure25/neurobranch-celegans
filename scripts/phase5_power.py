"""Approximate two-sided power for an event-level AUROC test (binormal)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(11)
n_pos, n_neg = 62, 188
for auc in (0.55, 0.60, 0.65, 0.70, 0.75):
    # Equal-variance binormal separation d = sqrt(2) * Phi^{-1}(AUC).
    from scipy.stats import norm

    d = np.sqrt(2) * norm.ppf(auc)
    hits = 0
    reps = 1500
    for _ in range(reps):
        scores = np.r_[rng.normal(d, 1, n_pos), rng.normal(0, 1, n_neg)]
        labels = np.r_[np.ones(n_pos), np.zeros(n_neg)]
        observed = roc_auc_score(labels, scores)
        # Approximate null SE from Hanley-McNeil at A=0.5.
        se0 = np.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))
        if abs(observed - 0.5) > 1.96 * se0:
            hits += 1
    print(f"AUC={auc:.2f} approximate power={hits / reps:.2f}")
