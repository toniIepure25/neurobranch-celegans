# Phase 13 results: projection-consistent reliability and event-level variance

**Status:** completed, **not adopted**. The decision is
`not_adopted_gate_failures`. This is a negative measurement-model result, not a
biological discovery.

> **Phase 14A follow-up.** The Phase 13 G6 comparison used a different one-event
> variance treatment from the Phase 11 reference. Phase 14A reproduced the
> reference and found it robust under a robust event variance (V1 0.3006 versus
> V0 0.3136; delta 0.0131). The Phase 13 negative decision stands because its SBC
> rank, null controls, and unrun PPC fail independently. See
> [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md) and correction C13.

**Protocol:** `research/phase13_protocol.md` (frozen before the full run)
**Deviations:** `research/phase13_deviations.md` (D1--D6, recorded before
interpretation)
**Branch:** `research/phase13-measurement`
**Base:** `5dc4407` (`v1.3.0`)
**Artifacts:** `reports/phase13/` (summary, per-target fits, SBC, nulls, prior
audit, manifest)

---

## 1. One-paragraph answer

The Phase 12R model failed because it combined two distinct approximations:
animal centering that left the noise variance unchanged, and a `1e-8` floor that
treated 79% of observed cells (mostly one-event cells) as almost noiseless.
Phase 13 separated them. Correcting only the centering (M1) barely moved the
point estimate (0.3138; ceiling 0.5602) and still failed calibration. Replacing
the floor with an event-level variance model (M2) moved the estimate to 0.2003
(ceiling 0.4475), showing that the old 0.304 was substantially inflated by the
floor. But M2 also failed the frozen gates: SBC coverage 0.9244 (gate
0.93--0.97), SBC rank KS p = 0.0031 (gate > 0.01), negative-control median
R(22) = 0.1485 and q95 = 0.4680, sign-flip median 0.1341 and q95 0.5796. The
held-out PPC was not run because the null controls had already failed. Both
candidates are rejected. **0.560 remains the only defensible descriptive
coverage-aware ceiling.** The M2 value 0.200 is a model-based sensitivity, not a
calibrated replacement.

---

## 2. Frozen gates

| Gate | Requirement | M2 observed | Result |
|---|---|---:|---|
| G1a SBC coverage | 0.93--0.97 | **0.9244** | **fail** |
| G1b SBC rank | KS p > 0.01 | **0.00306** | **fail** |
| G1c grid edge | max mass < 0.05 | 0.0121 | pass |
| G1d fit integrity | <= 5% failed/edge fits | 0 / 1,800 | pass |
| G1e one-event stratum | coverage 0.90--0.98 | 0.9231 | pass |
| G2 held-out PPC | coverage >= 0.90; MAE <= 0.10 | **not run** | **fail** |
| G3 negative control | median <= 0.05; q95 <= 0.15 | **0.1485; 0.4680** | **fail** |
| G3b sign-flip control | median <= 0.05; q95 <= 0.15 | **0.1341; 0.5796** | **fail** |
| G4 prior sensitivity | pooled R(22) range <= 0.05 | 0.00661 | pass |
| G5 event-variance recovery | coverage 0.90--0.98 | 0.9356 | pass |
| G6 Phase 11 agreement | within 0.05 of 0.314 | **0.2003 (diff 0.1137)** | **fail** |

Six of eleven gates failed. G1a is borderline but fails the frozen interval; it
is not relabelled as a pass. G2 is a protocol deviation: no PPC number is
invented, and the missing check counts as a failure.

---

## 3. Real-data estimates

All models use the same 126 eligible targets out of 259, coverage 0.80,
`min_events=1`, 2,500 posterior draws on an 80 x 80 grid, and target-specific
seeds derived from seed 12.

| Model | Pooled R(22) | Ceiling | Floor fraction | Event-variance median | Note |
|---|---:|---:|---:|---:|---|
| M0 Phase 12R reference | 0.3035 | 0.5509 | 0.7892 | — | historical control; not adoptable |
| M1 projection-scaled known `v` | 0.3138 | 0.5602 | 0.7892 | — | diagonally corrected; diagnostic only |
| M2 event-variance | 0.2003 | 0.4475 | 0.0000 | 0.6906 | primary candidate; rejected |

M0 reproduces the Phase 12R estimate exactly (0.3035306473). M1 is within 0.0002
of the Phase 11 moment estimate, so the centering correction alone is nearly
neutral for the point estimate. M2 is 0.1137 below 0.314 and therefore fails G6
even before the calibration failures.

The M2 event-variance median is 0.6906 on the standardized scale. 78.9% of
observed cells have one event; those cells now receive `v = omega_e^2`, not
`1e-8`. That is the mechanism of the estimate drop. It is stronger evidence than
the old hand-waved claim that the floor "might matter".

---

## 4. What the failure diagnoses

### 4.1 The floor was the dominant defect, not centering

The centering factor `1 - 1/R_a` changes `nu` modestly for typical profiles and
changes the pooled estimate by less than 0.01. The floor changes `nu` by several
orders of magnitude for 79% of cells and changes the estimate by 0.11. Any future
reliability analysis of this atlas must model one-event cells explicitly; a
floor is not neutral.

### 4.2 The event-variance model is better but still not calibrated

M2 passes event-variance recovery (0.9356) and prior sensitivity (0.0066), and it
removes the floor. It still fails SBC rank (p = 0.0031), SBC coverage (0.9244),
and both null controls. The rank failure means the posterior is not distribution-
calibrated even though its 95% interval covers at approximately the nominal rate
on average. The negative control is decisive: under within-animal responder
permutation, the model still reports median R(22) = 0.1485, so it manufactures
apparent between-target reliability from sampling noise. A real-data estimate of
0.2003 cannot be separated from that positive null bias with this model.

### 4.3 The null bias is not repaired by modeling event variance

Permutation preserves each animal's marginal values but destroys the shared
responder profile. A correctly specified model should estimate `sigma -> 0` and
R(22) -> 0. The observed 0.1485 median is a specification failure: the
half-normal prior and the additive Gaussian likelihood still prefer a nonzero
between-target variance. Possible repairs (not tested here, not adopted) are a
spike-and-slab prior on `sigma`, an explicit point mass at zero, or a
coverage/detection model that treats the observed mask as generated rather than
conditioned on. None may be claimed from this run.

### 4.4 M1 is not a fallback

M1's point estimate is close to 0.314, but its SBC rank also fails (KS
p = 2.14e-06) and it retains the floor. It is a diagnostic showing the centering
algebra was not the main driver. It cannot support an interval or ranking.

---

## 5. Sensitivity analyses (descriptive, not adoption evidence)

M2 pooled R(22):

| Coverage | Eligible targets | R(22) | Ceiling |
|---:|---:|---:|---:|
| 0.50 | 126 | 0.1471 | 0.3836 |
| 0.80 | 126 | 0.1973 | 0.4442 |
| 1.00 | 79 | 0.3691 | 0.6075 |

`min_events` at coverage 0.80:

| `min_events` | Eligible targets | R(22) | Ceiling |
|---:|---:|---:|---:|
| 1 | 126 | 0.1973 | 0.4442 |
| 2 | 36 | 0.2923 | 0.5406 |

The `min_events=2` estimate is closer to 0.314 but uses only 36 targets and a
different responder population. It is not a replacement for the primary
analysis, and it does not repair the negative control. Sensitivity rows use a
separate 1,200-draw fit and can differ slightly from the 2,500-draw primary
estimate (0.2003); the gate and decision use the primary configuration.

---

## 6. What is and is not claimed

**Supported (descriptive, bounded):**

- Single-animal and repeat-stimulation reliability remain below 0.5 under the
  Phase 9/10 estimators (unchanged; see `docs/CLAIMS.md`).
- The coverage-aware descriptive ceiling remains **0.560** under the Phase 11
  estimator.
- The Phase 12R model's 0.304 estimate is not a stable measurement limit: it is
  sensitive to the one-event variance treatment (M2 gives 0.200) and to
  `min_events` (0.292 with 36 targets).
- The centering projection alone is a small correction for this atlas; the
  variance floor is the larger problem.

**Not claimed:**

- No calibrated interval, target ranking, animal-budget guarantee, or
  "true reliability is 0.200" statement.
- No causal effect, digital twin, foundation model, or virtual-perturbation
  claim.
- No claim that M2 is the correct measurement model. It failed its own gates.
- No claim that the negative-control failure is biological; it is a model
  specification failure.

---

## 7. Reproduction

```bash
python scripts/phase13_measurement.py --out reports/phase13
python scripts/phase13_manifest.py --reports reports/phase13
python scripts/phase13_validate.py --reports reports/phase13
```

Quick mode (`--quick`) is a runtime/debug pilot and is never decision evidence.
The full run uses the frozen seeds 31/32/43/44/47 and the artifact manifest
pins every file hash. The input table is referenced by SHA-256 and is not
redistributed.

---

## 8. Next experiment implied by this result

The next model must make `sigma = 0` a credible point under the permutation null
before it is fit to real data. A spike-and-slab or zero-inflated between-target
prior is the minimal test; a coverage/detection generative model is the more
principled test. The negative-control median R(22) must be <= 0.05 in a
pre-registered run before any real-data posterior is interpreted. This is a new
protocol, not a Phase 13 retry. See `docs/NEXT_EXPERIMENTS.md`.
