# Phase 14 methods

This document describes the frozen Phase 14 analysis. The authoritative
thresholds are in
[phase14_protocol.md](https://github.com/neurobranch/neurobranch/blob/main/research/phase14_protocol.md);
this is an implementation and interpretation guide.

## 1. Input and eligibility

- Input: `reports/phase3/randi_trial_table_phase3.csv.gz`, SHA-256 in the
  manifest. Raw recordings are not redistributed.
- Response: `response_mean`.
- Animal unit: the Phase 11 `dataset_id` key.
- Coverage: 0.80; fold size 22; at least 6 animals; at least 3 responders.
- The 126 eligible targets are exactly the Phase 11 reference population.

## 2. Reference audit

`cell_summaries` returns one row per animal/target/responder with `n`, `mean`,
and the sample variance of the events. The historical estimator fills a
one-event cell with `var = 0`.

Phase 14 adds two keyword-only options:

- `one_event_variance`: replace `var` only when `n <= 1`;
- `event_variance`: replace `var` for every cell.

The returned `var` is a per-event variance. The cell-mean sampling variance is
`var / n`.

The primary audit uses `omega^2 = median(s^2)` over cells with `n >= 2` and
`s^2 > 0`. It is robust to the heavy-tailed raw cell-variance distribution.
V1h uses the same value for every cell; V2 uses `min_events=2`. The raw-scale
scaled-chi-square / HalfNormal plug-in is reported only as a failed diagnostic
(D1).

Bootstrap: animal clusters are resampled with replacement, 2,000 times, seed 61.
The per-target sufficient statistics are resampled rather than the full event
table, preserving the target masks and sampling variances.

## 3. Model M4

For target `t`, animal `a`, responder `r`:

```
z_t ~ Bernoulli(pi0), pi0 = 0.5
mu_r | z_t = 1 ~ N(0, sigma^2)
mu_r | z_t = 0 = 0
u_ar ~ N(0, tau^2)
e_ar ~ N(0, v_ar)
v_ar = omega^2 / n_ar
```

`omega^2` is the Phase 13 marginal event-variance posterior median. Profiles are
centered and the known variance is projection-scaled by `1 - 1/R_a`. The model
therefore remains on the same estimand as Phase 13.

Priors: `sigma ~ HalfNormal(1)` on the slab, `tau ~ HalfNormal(1)`. The spike is
an exact point mass at `sigma = 0`, not a narrow Gaussian.

### Computation

The slab is evaluated on an 80 x 80 grid of `(log sigma, log tau)`. The spike is
evaluated on the `tau` grid at `sigma = 0`. Log evidence for the two components is
combined with `logaddexp`; posterior spike mass is

```
P(z=0 | y) = pi0 * L_spike / (pi0 * L_spike + (1-pi0) * L_slab).
```

Posterior draws include exactly zero `sigma` values. Reliability is

```
R(n) = sigma^2 / (sigma^2 + (tau^2 + nu) / n),
```

so the spike produces `R(n) = 0` exactly. The real-data summary is the median of
per-target posterior medians, matching Phase 11/13.

## 4. Calibration

### SBC

1,000 replicates x 3 real-target templates = 3,000 fits. Truth:
`z ~ Bernoulli(0.5)`, `sigma ~ HalfNormal(1)` when `z=1`, `tau ~ HalfNormal(1)`,
and `omega` from its prior. The mask, event counts and sampling variances are
preserved. Rank ties at `R=0` use the randomized mid-rank; a KS test with
`p > 0.01` is required.

### Held-out PPC

20 animal-level splits, 20% held out, up to 40 targets per split, 1,200 draws,
60 x 60 grid, seed 41. The training fit is reused for the held-out profile. The
observed and predicted profiles are centered identically; coverage and median
absolute error are computed at the target level. A PPC with fewer than 100
records is a failure.

### Null controls

Within-animal responder permutation and sign flip, 100 targets, 1,200 draws,
50 x 50 grid, seeds 43 and 44. The null preserves event counts and sampling
variances. Both median and q95 must pass; a zero median alone is insufficient.

### Prior sensitivity

Slab scales 0.5/1.0/2.0 and `pi0` 0.25/0.5/0.75, 1,200 draws, 50 x 50 grid, seed
47. The range of the pooled median must be at most 0.05.

## 5. Gate logic

All gates are conjunctive. `n_gates_failed` counts failed gates; a model is
adopted only when it is zero. There is no post hoc threshold relaxation. A
missing PPC or missing SBC is a failure. The validator recomputes the decision
from the gate object, checks the PPC artifact, requires at least 3,000 SBC fits,
and verifies the artifact manifest.

## 6. Reproducibility

- Seed, configuration and input hash are written to `summary.json`.
- Text artifact hashes are LF-normalized.
- `phase14_manifest.py` records code/worktree state outside `reports/`.
- `phase14_validate.py` fails on schema violations, gate/decision mismatch,
  missing PPC, insufficient SBC, artifact hash mismatch, stale checksum, or a
  dirty code worktree.

## 7. What the method cannot do

The model is a measurement model. It does not identify a causal effect, a
mechanism, a digital twin, or a generalizable target ranking. Passing SBC would
mean the interval is calibrated under the simulation model, not that the
simulation model is biologically true.
