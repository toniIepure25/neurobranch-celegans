# Master Review: Research History and Claim Lineage

**Purpose:** reconstruct how the project arrived at the Kato common-state claim, including the negative results, corrections and dead ends that a reader of the final memo would not see.  
**Method:** Git history, committed reports, experiment registry, correction log, and artifact manifests. No history was rewritten.  
**Machine-readable timeline:** [`reports/master_review/history_timeline.csv`](../reports/master_review/history_timeline.csv).

---

## 1. Why history belongs in a scientific audit

A result is not just a number. It is the endpoint of a sequence of choices: which dataset, which split, which baseline, which metric, which correction, which claim. If the sequence included test reuse, a changed sign convention, a withdrawn ceiling, or a baseline that was later found to be weak, the final number may be numerically correct while the claim attached to it is too strong.

This project has an unusually complete record of its own reversals. That is a strength. It is also a warning: the final documents are not a clean pre-registered story. They are a layered record in which later documents correct earlier ones, sometimes without the earlier ones being changed.

The audit therefore separates:

- **what was run** (artifacts, commands, manifests);
- **what was claimed at the time** (historical reports);
- **what is claimed now** (final audit and master review);
- **what changed** (corrections, sign convention, baseline definitions).

---

## 2. Phase summary

| Phase | Dates | Anchor | Question | Outcome | Status |
|---|---|---|---|---|---|
| P0 | 2026-10-22/23 | `f64e761` | Build a reproducible platform | Blocked suite, forward stress, NC-MCM/BunDLe artifact checks | Foundation |
| P1 | 2026-10-23 | `5b32c9c..` | Are inherited claims sound? | Branching null reinterpreted; intervention/BunDLe caveats | Audit |
| P2 | 2026-10-23 | Phase 2 branch | Do named-neuron interventions validate a target? | No validated target; winner’s curse exposed | Negative |
| P3 | 2026-10-23 | Phase 3 branch | Prospective behavioural information in an independent dataset? | Baskoylu 27 animals: neural increment inconclusive | Negative/inconclusive |
| P4 | 2026-10-24 | Phase 4 branch | Turn-direction prediction? | Kramer 56 animals: ΔAUROC +0.0123, CI crosses zero | Null |
| P5 | 2026-10-24/25 | Phase 5 branch | Reconstruct SAAV? | Exact reproduction blocked; partial reconstruction null | Blocked/partial |
| P6 | 2026-10-31 | Phase 6 branch | Autonomous simulator fidelity? | G2/G3/G4 fail; G5 interface only | Negative |
| P7 | 2026-11-01 | Phase 7 | Active probing? | No benchmark superiority; normalised scores withdrawn | Withdrawn |
| P8 | 2026-11-01 | Phase 8 | Adaptive twin pair advantage? | Underpowered (MDE 0.0587 vs SESOI 0.020) | Underpowered |
| P9 | 2026-11-01 | Phase 9 | Measurement ceiling? | Reliability estimator replaced; ceiling corrected | Corrected |
| P10 | 2026-10-29 | Phase 10 | Metric-aligned ceiling/coverage? | Coverage study negative; ceiling descriptive | Negative |
| P11 | 2026-10-29 | Phase 11 | Publish 0.560 pooled ceiling? | Withdrawn; extrapolation fails | Withdrawn |
| P12 | 2026-10-30 | Phase 12/12R | Hierarchical calibration? | SBC coverage 0.923; interval not calibrated | Failed |
| P13 | 2026-10-31 | Phase 13 | Stable measurement variant? | `R_obs` 0.2606; ranking/extrapolation fail | Descriptive |
| P14 | 2026-10-31 | Phase 14 | ZINB rescue? | Rejected, 8/12 gates fail | Rejected |
| P15 | 2026-11-01 | Phase 15 | Defensible reliability estimand? | `R_obs` bounded; no ranking | Bounded |
| K1 | 2026-11-02 | `23b0ed1` | Predictive cross-neuron information in Kato? | B3 beats B1; operator unstable | Exploratory |
| K2 | 2026-11-02/03 | `358107a` | Survives own history + ten factors? | +0.2644; 4/5 sessions; H1 reverses | Exploratory |
| K3 | 2026-11-03 | `9f4f833..96a8dd3` | End-to-end reproduction? | Clean rerun passes; G4 confirmatory not established | Reproduced/exploratory |
| M1 | 2026-11-03 | master review | Are identity, bounds, statistics defensible? | Bounds pass; animal-level interval includes zero | Methodological review |

---

## 3. Pre-Kato phase: accumulating negative evidence

### 3.1 P0: the platform and its first temptation

The repository began with a full pipeline: Kato loading, blocked and forward splits, VAR/MLP/GRU/latent models, behaviour probes, branching analysis, virtual perturbation, NC-MCM and BunDLe-Net artifact reproduction. It had 105 blocked runs and 35 forward-stress runs.

The first scientific temptation was to treat **interpolation** as prediction. The blocked split trains on blocks 0, 1, 3 and tests on block 2: the model sees the future on both sides of the test block. That is a legitimate model-class comparison under a stationarity assumption, but it is not a forecasting benchmark. The documentation eventually labelled it clearly; the early framing was less careful.

The second temptation was to treat **one-step R²** as simulator validity. A model can have good one-step R² and fail catastrophically when rolled forward. Phase 6 later made this explicit: the selected model had validation one-step R² `0.7134`, free-running test R² `-0.5098`, and OOD fraction `0.79`.

**Lesson:** interpolation and one-step accuracy are not free-running skill.

### 3.2 P1/P2: branching and intervention

The inherited branching analysis reported `p = 1.0`. That looked like “no branching,” but the test permuted outcome labels across candidate frames and used a one-sided `observed >= null` rule. It was a valid directional test for excess local diversity, not a test of the opposite direction, and it did not preserve autocorrelation. Phase 2 separated “no excess branching” from “no branching exists.”

The inherited intervention analysis had a more serious problem: the candidate set was literature-motivated after inspecting the same worm, and the control set was only 19 non-candidate cells. Winner’s curse, model-seed variability, amplitude and architecture were not jointly evaluated. Phase 2 concluded that no named neuron was a validated target. This is the origin of the standing label:

```text
MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.
```

**Lesson:** a model’s sensitivity to a virtual push is not a biological effect.

### 3.3 P3: the independent prospective benchmark

Phase 3 used Baskoylu & Flavell (2026), 27 animals, 43,200 frames, 1,162 validated reversals. The prespecified task was reversal onset in `(t, t+H]` with animal-disjoint leave-one-animal-out evaluation and a proper score.

At the primary 6 s horizon:

```text
behaviour log loss        0.5927
behaviour + neural        0.5856
Δ                        -0.00701
95% CI                   [-0.01483, +0.00053]
animals improved         20/27
```

The criterion required the CI upper bound below zero and at least 70% of animals improved. It failed. The result was **inconclusive/negative**, not proof of absence.

A train-only PCA(10) secondary analysis had a CI excluding zero (`-0.00841 [-0.01475, -0.00220]`), but it failed robustness: adding neural features to a quadratic behavioural state gave `-0.00524 [-0.01334, +0.00301]`; the identified-neuron representation was weaker; the within-animal forward task was worse with neural activity. The project correctly did not promote the PCA result.

**Lesson:** a secondary representation that survives one CI but fails a stronger behavioural control is not a finding.

### 3.4 P4: turn direction

Phase 4 used Kramer et al. (2026), 73 recordings, 56 WT animals, 945 events for the primary turn-direction set. The primary model was logistic regression with animal-disjoint folds.

```text
behaviour + population AUROC   0.608
behaviour AUROC                0.595
Δ AUROC                        +0.0123
95% CI                         [-0.0166, +0.0425]
p(Δ≤0)                         0.210
```

No incremental evidence. SAAV, circuit and identified-neuron contrasts were null or worse. The independent 6 s replication also failed its criterion (`Δ = -0.00570`, CI `[-0.01133, +0.00047]`, 22/32 improved, below 70%).

**Lesson:** a small positive point estimate with a CI crossing zero is not an increment.

### 3.5 P5: the missing-input boundary

Phase 5 attempted to reconstruct the SAAV decoder from Kramer et al. The official code required four aligned CSVs (`SAAV_dorsal.csv`, `SAAV_ventral.csv`, `HC_dorsal.csv`, `HC_ventral.csv`) that were not in the repository or Dryad. Exact reproduction was `BLOCKED_MISSING_INPUTS`, not “difficult.”

The partial reconstruction found no positive SAAV increment in log loss in any regime. The external cohort showed adverse transfer (AUROC 0.33 vs 0.56 for behaviour), interpreted as distribution shift and small-sample instability, not harm.

**Lesson:** missing inputs are a hard boundary. A partial reconstruction is not a reproduction, and a null partial reconstruction is not a refutation of the published method.

### 3.6 P6: the simulator gate

Phase 6 corrected the history/evaluation contract and selected models on validation block 4. The test block had already been inspected in Phase 5, so Phase 6 treated it as exploratory.

Gates:

```text
G1 functionality                 pass
G2 autonomous skill              fail (3/5 vs frozen Phase 5 VAR)
G3 distributional fidelity       fail (0/5)
G4 stochastic branching          fail
G5 support-constrained perturb.  pass
G6 robust target specificity     not_evaluable (one family)
G7 biological validation         not_evaluable
```

No learned family beat the frozen VAR in free-running skill. The selected `mlp_residual` had the best validation one-step R² and the worst free-running test R².

**Lesson:** model selection on one-step validation can select the worst free-running model.

---

## 4. The measurement detour: P7–P15

The project then shifted from dynamics to measurement reliability. This detour is important because it produced the intellectual discipline that later shaped the Kato audit: estimands, support, calibration, and the difference between a descriptive estimate and a guarantee.

### 4.1 P7–P9: ceilings and corrections

Phase 7 normalised model scores by a split-half ceiling. Some scores exceeded 1. The denominator had been truncated at zero. A truncated denominator is not a ceiling. Normalised scores were withdrawn.

Phase 8 reported a pair-level advantage with an interval excluding zero (`[+0.0080, +0.0901]`). Phase 9 showed the reconstructed MDE was `0.0587` against a smallest effect of interest of `0.020`; the claim was **underpowered**, not supported. A CI excluding zero can coexist with a design that cannot resolve the effect of interest.

Phase 9 also replaced the split-half Spearman-Brown ceiling (`0.152`) with variance-corrected estimates. Correction C2 withdrew “CRO reaches 64% of the ceiling”: the corrected single-animal ceiling was `0.473`, so `0.0984` was about `20.8%`, not 64%.

**Lesson:** an estimator choice can change a ceiling by a factor of three. A ratio above 1 is a diagnostic of a bad denominator.

### 4.2 P10–P11: coverage and extrapolation

Phase 10 built a metric-aligned ceiling and a synthetic coverage study. The coverage study was negative. Phase 11 reported a pooled ceiling of `0.560` from `sqrt(R(22))`, with `R(22)=0.3136`.

The problem: only 39 of 126 targets had at least 22 animals. The other 87 values were extrapolated. Phase 11 was withdrawn.

**Lesson:** do not extrapolate a reliability bound outside support.

### 4.3 P12–P15: calibration and estimands

Phase 12 attempted a hierarchical measurement model with a calibrated posterior interval. It reproduced the point estimate but failed calibration:

```text
SBC 22-animal coverage        0.923 (gate 0.93–0.97)
rank uniformity KS p          1.4e-06
held-out predictive coverage  0.896 (gate >= 0.90)
median absolute error         0.191 (gate <= 0.10)
```

Phase 13/15 settled on three estimands:

```text
R_obs  = 0.2606  [0.1782, 0.3430]   126 targets, 0 extrapolated
R_22   = 0.3006  [0.2087, 0.3924]   126 targets, 87 extrapolated
R_id   = 0.3501  [0.2020, 0.4982]   39 targets, 0 extrapolated
```

`R_obs` is descriptive. `R_22` is a design counterfactual. `R_id` is a different population. Extrapolation failed (median absolute error `0.204`, bias `+0.038`), and ranking failed (median Spearman `0.126`, top-10 Jaccard `0.176`). No ranking was published.

Phase 14’s ZINB model was rejected: `M4` median `0.0543`, 8/12 gates failed, prior-dominated.

**Lesson:** a point estimate matching is not interval calibration. Report the estimand and its population.

---

## 5. The Kato turn: K1–K3

### 5.1 K1: predictive cross-neuron information

The Kato interaction audit asked whether a full lag-one population operator (B3) forecasts H20 joint trajectories better than an independent per-neuron model (B1) with matched innovations.

The first result was positive on energy:

```text
B3 vs B1 energy    +0.371  [0.128, 0.530]
B3 better          60.6% of landmarks
B3 vs B4r energy   +0.547  [0.355, 0.698]
B3 vs B4r MSE      +0.428  [0.191, 0.621]
```

But the operator was not stable:

```text
spectral radius segment CV          0.0015
off-diagonal coefficient correlation 0.052
non-normality                        1.47–2.70
transient amplification              3.6–4.7x
```

The state-transition law was worse than a training-only Markov null (`TV 0.053` vs `0.031`). Virtual perturbations disagreed across fits (median cosine `0.556`; only AVEL exceeded `0.80`; some sign reversals). The structural gate G7 failed; biological G8 was `NOT_EVALUABLE`.

**Lesson:** prediction does not identify coupling. A stable spectral radius does not imply a stable operator or a stable transient.

### 5.2 K2: the common-factor control

The next question was stronger: does the population advantage survive each neuron’s own history, a local drift term, and ten training-only common factors?

The matched ladder at H20 PCA-10 energy:

```text
B0 persistence              3.6570
B1m own AR (200)            3.0459
B2 own AR + drift           3.5685
B4m own AR (500)            3.2397
B5-10 own + ten factors     2.9646
B3 full population VAR      2.7001
```

The primary contrast:

```text
B5 - B3 = +0.2644
block-5 interval [+0.0750, +0.4031]
B3 better at 64.0%
```

Four of five sessions positive; session 2 negative (`-0.1931`). H1 negative (`-0.0637`), H5 `+0.0835`, H10 `+0.1471`, H20 `+0.2644`. PCA-10 MSE `+0.0545` with interval crossing zero. Neuron-space energy `+0.1617` with interval crossing zero.

The ordered decomposition:

```text
B4m - B5   = +0.2751  (51.0%)
B5 - B3    = +0.2644  (49.0%)
B4m - B3   = +0.5395
```

This is arithmetic, not a unique partition. With B1m as baseline the shares are `23.5/76.5`; with B0 they are `72.4/27.6`.

**Lesson:** common factors explain about half under one ordering. The split is not a structural estimate.

### 5.3 K3: reproduction and the sign ambiguity

The final audit ran a clean end-to-end rerun:

```text
status                  FULL_END_TO_END_REPRODUCTION_PASSED
runtime                 2,005.9 s
stderr                  empty
max absolute difference 4.44e-16 across 17 artifacts
max relative difference 9.84e-16
```

That is a genuine numerical reproduction. It is not independent scientific confirmation.

The sign convention was another matter. The lower-is-better target was physically clear, but the only committed protocol was added with the results, contained an explicit “sign-convention clarification,” and had no independent pre-evaluation commit. The final audit therefore downgraded G4:

```text
numerical pass under corrected sign
confirmatory status NOT_ESTABLISHED
evidential tier SUPPORTED_EXPLORATORY
```

The final decision remained `READY_FOR_SUPERVISOR_REVIEW`, explicitly meaning “ready for supervisory judgement of an exploratory result, not ready for publication as a confirmed mechanism.”

**Lesson:** reproduction is not confirmation. A protocol written after seeing the result cannot carry the same evidential weight as a pre-registered one.

---

## 6. Claim lineage

| Early claim | Later correction | Current status |
|---|---|---|
| “Branching p=1.0 means no branching” | Directional test only; does not test opposite direction | `UNSUPPORTED` as stated |
| “Named neuron X is a validated intervention target” | Winner’s curse; no biological validation | `NOT_EVALUABLE` |
| “Neural activity adds prospective behavioural information” | Animal-disjoint CI crosses zero | `UNSUPPORTED` |
| “Population activity predicts turn direction” | ΔAUROC CI crosses zero | `UNSUPPORTED` |
| “SAAV reconstructed” | Missing inputs; partial reconstruction only | `BLOCKED/PARTIAL` |
| “Simulator captures dynamics” | Free-running gates fail | `UNSUPPORTED` |
| “CRO reaches 64% of ceiling” | Corrected ceiling; 20.8% | `INVALIDATED` |
| “Pair advantage supported” | Underpowered | `UNDERPOWERED` |
| “Pooled ceiling 0.560” | Extrapolation outside support | `INVALIDATED` |
| “Calibrated hierarchical interval” | SBC/coverage fail | `UNSUPPORTED` |
| “B3 identifies cross-neuron interactions” | Operator unstable; structural gate fails | `UNSUPPORTED` |
| “G4 fully prespecified PASS” | Sign clarified with results | `INVALIDATED` as confirmatory |
| “51/49 unique partition” | Ordered decomposition only | `INVALIDATED` as unique |
| “B3 improves forecast accuracy” | MSE intervals cross zero | `UNSUPPORTED` |
| “Five repeated sessions, not animals” | Five animals inferred; one cohort | `INVALIDATED` as stated |
| “175 independent landmarks” | Overlapping windows | `INVALIDATED` |
| “Ready for publication” | Exploratory, reused test, no intervention | `UNSUPPORTED` |

---

## 7. What the history reveals about the process

### Strengths

- The project repeatedly corrected itself, including unfavourable corrections.
- It preserved negative results and failed gates.
- It distinguished artifact reproduction from scientific reproduction.
- It built calibration and coverage checks that later exposed its own overclaims.
- It kept a written record of test reuse and sign ambiguity.

### Weaknesses

- Several corrections were additive: the old document still says the wrong thing.
- The final memo can be read without the correction log.
- The phrase “five repeated sessions” was repeated across many documents after the publication described five animals.
- The primary interval is easy to overread as animal-level.
- The energy result is easy to overread as mean accuracy.
- The 51/49 split is easy to overread as a structural partition.

### The central lesson

The project’s numerical pipeline is more mature than its claim language. The fix is not a new model. The fix is a disciplined mapping from each number to the strongest sentence it supports, with the evidential unit and the alternative explanations stated.

---

## 8. Unresolved historical questions

1. Did any earlier phase use the test tails to choose the 500-frame window, the ridge, the factor count, or the B5 specification? The final protocol says no factor count was selected on test, but the test block had already been inspected in earlier phases.
2. Was the sign convention ever written before the results in a separate commit? The audit found no independent pre-evaluation commit.
3. Why does `docs/DATA_PROVENANCE.md` contain a size that matches neither the local file nor OSF? It may be a copied size from a different artifact; the origin is not documented.
4. Why do multiple documents call the five recordings “repeated sessions” when the publication and secondary sources describe five worms? The correction is clear; the origin of the error is not.
5. Was the B5 factor basis ever refit locally in an earlier version? The final code uses a globally fixed training PCA; earlier reports may have used a different construction.

These are historical uncertainties, not active computational defects. They matter because they limit how much confirmatory weight the final protocol can carry.

---

## 9. Timeline artifacts

The machine-readable timeline is [`reports/master_review/history_timeline.csv`](../reports/master_review/history_timeline.csv). It is a curated phase-level reconstruction, not a commit-by-commit log. The full Git history remains the authoritative record.

The correction register is [`reports/master_review/corrections.json`](../reports/master_review/corrections.json). The claims ledger is [`reports/master_review/claims_ledger.json`](../reports/master_review/claims_ledger.json).
