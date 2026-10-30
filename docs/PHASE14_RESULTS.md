# Phase 14 results: reference audit and zero-inflated reliability

**Status:** negative result. Phase 14 did **not** adopt a calibrated reliability
model. The Phase 14A reference audit supports the historical `0.560` descriptive
ceiling under a robust one-event variance treatment, but the Phase 14B
spike-and-slab model fails eight of twelve frozen gates. No calibrated interval,
target ranking, animal-budget guarantee, causal effect, or digital-twin claim is
published.

**Run date:** 2026-10-31
**Model:** `m4_spike_slab`
**Decision:** `not_adopted_gate_failures`
**Artifacts:** `reports/phase14/`
**Protocol:** [phase14_protocol.md](https://github.com/neurobranch/neurobranch/blob/main/research/phase14_protocol.md)
**Deviations:** [phase14_deviations.md](https://github.com/neurobranch/neurobranch/blob/main/research/phase14_deviations.md)

## 1. Why Phase 14 was run

Phase 13 rejected two repairs of the Phase 12R model, but its reference was not
like-for-like. Phase 11's `cell_summaries` sets the variance of a one-event cell
to zero; Phase 13 M2 uses `omega^2 / n`. The frozen Phase 13 G6 gate therefore
compared an M2 fit with a reference built under a different variance treatment.

Phase 14 separated two questions:

1. Is the reference `0.314 / 0.560` robust to a nonzero one-event variance?
2. Does an exact zero/spike-and-slab model fix the positive permutation-null bias
   and pass calibration?

## 2. Phase 14A reference audit

The primary audit replaces the one-event zero with a robust global event variance
`omega^2 = 1.7629` (median of observed within-cell sample variances for cells
with `n >= 2`). This was frozen after a numerical pilot showed that a raw-scale
scaled-chi-square plug-in saturates at the grid edge because cell variances are
extremely heavy-tailed. The amendment is recorded as D1 before the full run.

| Treatment | One-event variance | `n >= 2` variance | Targets | R(22) | Ceiling |
|---|---|---|---:|---:|---:|
| V0 historical | 0 | sample | 126 | 0.3136077 | 0.5600069 |
| V1 primary | `omega^2 / n` | sample | 126 | 0.3005503 | 0.5482247 |
| V1h homogeneity | `omega^2 / n` | `omega^2 / n` | 126 | 0.3138911 | 0.5602599 |
| V2 filter | dropped | sample | 36 | 0.1753713 | 0.4187736 |

- **A1 passed:** V0 reproduces the Phase 11 values exactly (`1e-6`).
- **A2 passed:** `|V1 - V0| = 0.01306 < 0.05`. The reference is robust to a
  nonzero one-event variance.
- Animal-cluster bootstrap, 2,000 resamples: V0 95% interval `[0.5044, 0.6500]`;
  V1 `[0.4902, 0.6393]`.
- The `min_events=2` filter is **not** a variance-treatment sensitivity: it
  changes the target population from 126 to 36 and lowers the estimate to
  0.1754. The `0.560` bound applies to the coverage-aware 126-target population,
  not to a two-event subset.

The audit does not validate `0.560` as a calibrated interval. It remains a
descriptive attenuation bound under the Phase 11 estimator.

## 3. Phase 14B model and real-data result

Model:

```
z_t ~ Bernoulli(0.5)
mu_r | z_t = 1 ~ N(0, sigma^2)
mu_r | z_t = 0 = 0
u_ar ~ N(0, tau^2)
e_ar ~ N(0, omega^2 / n_ar)
sigma, tau ~ HalfNormal(1)
```

The spike is an exact point mass at `sigma = 0`. The fit uses an 80 x 80
`(log sigma, log tau)` grid, 2,500 draws, seed 12, coverage 0.80, and the Phase 13
projection scaling.

Real-data fit over the same 126 eligible targets:

| Quantity | Value |
|---|---:|
| Median pooled R(22) | **0.0543** |
| Median pooled ceiling | 0.2329 |
| Median single-animal R(1) | 0.0026 |
| Median posterior P(z=0) | 0.3235 |
| Mean posterior P(z=0) | 0.2887 |
| Targets with P(z=0) >= 0.5 | 6.35% |
| Event-variance median | 0.6906 |
| Maximum grid edge mass | 0.00559 |

The real-data median is far below both the Phase 13 M2 value (0.2003) and the
Phase 14A V1 reference (0.3006). This is not evidence that the true reliability is
0.054; it is evidence that the model's posterior is prior-dominated and not
calibrated.

## 4. Frozen gates

| Gate | Requirement | Observed | Status |
|---|---|---|---:|---|
| G1a SBC coverage | 0.93--0.97 | 0.9673 | pass |
| G1b SBC rank | KS p > 0.01 | 3.35e-192 | **fail** |
| G1c grid edge | < 0.05 | 0.0810 | **fail** |
| G1d fit integrity | <= 5% failures | 0 / 3,000 | pass |
| G1e one-event stratum | 0.90--0.98 | 0.9670 (n=5,870) | pass |
| G2 held-out PPC | coverage >= 0.90; MAE <= 0.10 | 0.7697; 0.3998 (n=304) | **fail** |
| G3 negative control | median <= 0.05; q95 <= 0.15 | 0.0000; 0.3535 | **fail** |
| G3b sign-flip | median <= 0.05; q95 <= 0.15 | 0.0000; 0.4509 | **fail** |
| G4 prior sensitivity | range <= 0.05 | 0.1220 | **fail** |
| G5 event-variance recovery | 0.90--0.98 | 0.9357 (n=3,000) | pass |
| G6 reference agreement | within 0.05 of V1 | 0.2463 away | **fail** |
| G7 spike calibration | median P(z=0 \| z=0) >= 0.80 | 0.3561 (n=1,503) | **fail** |

**Eight of twelve gates failed.** The model is not adopted.

### What the failures mean

- **Rank non-uniformity:** SBC ranks are concentrated below 0.5
  (mean 0.368), so the posterior is systematically shifted relative to the
  generative truth even though equal-tailed coverage is 0.967.
- **Grid-edge mass:** 8.1% of SBC fits put mass on the slab boundary, above the
  frozen 5% integrity threshold.
- **PPC:** held-out profiles are predicted with too little accuracy; observed
  correlations fall outside the predictive interval in 23% of target-level
  records.
- **Null controls:** medians are exactly zero because the spike is selected, but
  the upper tails remain too heavy (q95 0.35 and 0.45). A median-only pass would
  hide false positives on a minority of null targets.
- **Prior sensitivity:** changing `pi0` from 0.25 to 0.75 moves median R(22) from
  0.1220 to 0.0, a range of 0.1220. The result is prior-dominated.
- **Reference gap:** the model's 0.0543 is nowhere near the audited 0.3006.
- **Spike calibration:** when the truth is exactly zero, the median posterior
  spike probability is only 0.356, far below 0.80. The spike is not identified
  well enough to support a calibrated zero claim.

## 5. Decision and claim boundary

The decision is `not_adopted_gate_failures`. The exact zero/spike-and-slab model
is rejected as a calibrated measurement model on this atlas.

Retained:

- The Phase 11 descriptive ceiling `0.560` for the 126-target, 80%-coverage
  population, now with a robustness audit.
- The Phase 13 statement that the `1e-8` floor inflated the earlier estimate.
- Single-animal and repeat reliability below 0.5 as bounded descriptive claims.

Not supported:

- A calibrated pooled interval at any `n`.
- A target ranking or “most reliable neuron” list.
- An animal-budget guarantee.
- A causal effect of any neuron or stimulation.
- A digital twin, foundation model, or virtual-perturbation claim.
- Transport to another dataset or preparation.

## 6. What would be needed next

A model that passes all gates would need a better treatment of the
zero/nonzero decision and the prior scale, and an independently validated
held-out predictive check. That is a new protocol, not a Phase 14 retry with a
different `pi0` or grid. Candidate directions are recorded in
[NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md).

The negative result is useful: a point mass at zero alone does not repair the
null bias, and a median null of zero can coexist with a heavy false-positive
tail. Reporting only the median would have been misleading.
