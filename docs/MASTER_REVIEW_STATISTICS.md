# Master Review: Statistical Inference and Capacity

**Primary contrast:** `B5-10` H20 PCA-10 energy minus `B3` H20 PCA-10 energy.  
**Positive value:** B3 has lower energy (better).  
**Frozen sample:** 175 overlapping landmarks from five Kato `WT_NoStim` recordings.

---

## 1. The estimand

The analysis estimates a **paired, within-landmark, within-recording association**:

```text
Δ = E[ B5_energy - B3_energy | these five recordings, these landmarks ]
```

It does **not** estimate:

- the average effect in a new cohort of animals;
- the effect of a biological intervention;
- the effect of “cross-neuron coupling”;
- a population-average forecast improvement.

The distinction is not pedantic. The 175 landmarks are overlapping windows, and the five recordings are one laboratory cohort. The estimand is conditional on both.

The observed value is:

```text
Δ = +0.2644379926683563
```

with 64.0% of paired landmarks positive.

---

## 2. Effect size

| Quantity | Energy | MSE |
|---|---:|---:|
| Mean paired difference | `+0.2644` | `+0.0545` |
| SD of paired difference | `0.9777` | `1.2307` |
| Descriptive paired `dz` | `0.270` | `0.044` |
| Median | `+0.1860` | — |
| IQR | `[-0.1891, +0.8044]` | — |
| Fraction positive | `0.640` | `0.600` |

The energy effect is small-to-moderate by conventional standards (`dz ≈ 0.27`). The MSE effect is negligible (`dz ≈ 0.04`). The median energy advantage is smaller than the mean, and the IQR crosses zero: the effect is not a uniform shift.

The energy distribution is right-skewed: a subset of landmarks has large positive differences, while many are near zero or negative. This is consistent with heterogeneity, not a constant effect.

---

## 3. Dependence: 175 is not 175

### 3.1 Overlapping windows

Landmarks are 10 frames apart. Forecasts cover 20 frames at H20 and up to 100 frames at H100. Adjacent landmarks share:

- most of their history;
- most of their forecast trajectory;
- many of their target frames;
- the same fitted model family and often the same local fit window.

The approximate number of non-overlapping H100 windows is:

| Session | Landmarks | Test span | Non-overlapping H100 proxies | Ratio |
|---:|---:|---:|---:|---:|
| 0 | 35 | 440 | 5 | 7.0 |
| 1 | 35 | 440 | 5 | 7.0 |
| 2 | 34 | 430 | 5 | 6.8 |
| 3 | 38 | 470 | 5 | 7.6 |
| 4 | 33 | 420 | 5 | 6.6 |
| **Total** | **175** | | **25** | **7.0** |

At H20, the overlap is smaller: roughly 22 non-overlapping windows per session, about 110 total. But the H100 window is the evaluation support, and the forecast horizons are nested.

The `175` count is a count of **evaluation landmarks**, not independent biological replicates. Treating it as `n=175` inflates precision by roughly `sqrt(7) ≈ 2.6` relative to non-overlapping H100 windows, and much more relative to five animals.

### 3.2 Within-session autocorrelation

The lag-1 autocorrelation of the paired difference ranges from `0.179` to `0.597` across sessions. Lag-2 and lag-3 are non-zero in several sessions. The moving-block bootstrap attempts to respect this, but a 5-frame block is short relative to a 20-frame forecast overlap.

Subsampling every 2nd, 3rd and 5th landmark gives means `+0.208`, `+0.260`, `+0.401` on 89, 60 and 36 landmarks. The direction is stable; the magnitude is not. These are dependence-reducing sensitivity analyses, not independent tests. The every-5th estimate is larger, so it must not be presented as confirmation.

### 3.3 The five recordings

The recordings are the biological units. They are five animals (inferred), one cohort. A bootstrap that resamples landmarks within recordings answers:

> How variable is the estimate within these five recordings?

It does not answer:

> Would the sign replicate in another five animals?

That requires resampling recordings.

---

## 4. Intervals: three different targets

| Target | Construction | Mean | 95% interval | Includes zero? |
|---|---|---:|---|---|
| Conditional on five recordings | moving-block within session, equal pooling, block 5 | `+0.2644` | `[+0.0750, +0.4031]` | no |
| Conditional, block 20 | same, block 20 | `+0.2644` | `[+0.0921, +0.2939]` | no |
| Generalization across recordings | resample recordings, then blocks, block 5 | `+0.2352` | `[-0.0408, +0.4793]` | **yes** |
| Generalization across recordings, block 20 | same, block 20 | `+0.1897` | `[-0.1307, +0.4403]` | **yes** |
| Equal-weight session means | t interval over five means | `+0.2617` | `[-0.0641, +0.5874]` | **yes** |

The first two are the final audit’s primary and sensitivity intervals. The third and fourth are the master-review cluster bootstrap. The fifth is a simple check that ignores within-session precision.

The correct reading:

- **Within these five recordings**, the conditional interval excludes zero at block 5.
- **Across recordings**, the interval includes zero. Five units are too few to resolve an effect of this size.
- The block-20 conditional interval is narrower than block 5 because it resamples only two blocks per session; narrower is not more precise. The final audit says this explicitly.

### 4.1 Sign test

Session means:

| Session | Mean | B3 better fraction |
|---:|---:|---:|
| 0 | `+0.3003` | 0.686 |
| 1 | `+0.3292` | 0.686 |
| 2 | `-0.1931` | 0.500 |
| 3 | `+0.4075` | 0.579 |
| 4 | `+0.4645` | 0.758 |

Four of five positive. One-sided sign test `p = 0.1875`; two-sided `p = 0.375`. With five sessions, only 5/5 would reach `p = 0.03125`. The design cannot produce a significant sign test with one reversal.

### 4.2 Rough detectable effect

The session-mean SD is `0.2623`; the standard error is `0.1173`; the 95% t margin with 4 df is `0.3257`. A rough t-based smallest detectable mean (two-sided, 80% power, same SD) is larger, around `0.45–0.50` energy units. The observed `0.26` is below the two-sided detection threshold.

This is not a formal power calculation for the actual mixed design. It is a sanity check: **five sessions cannot resolve a small effect**, and the observed effect is small.

---

## 5. Horizon dependence

| Horizon | Mean `B5-B3` | Conditional block-5 95% interval | Direction |
|---:|---:|---|---|
| 1 | `-0.0637` | `[-0.0967, -0.0311]` | B5 better |
| 5 | `+0.0835` | `[+0.0339, +0.1145]` | B3 better |
| 10 | `+0.1471` | `[+0.0629, +0.2037]` | B3 better |
| 20 | `+0.2644` | `[+0.0750, +0.4031]` | B3 better |

The sign reverses at H1. The advantage grows with horizon. This is consistent with a multi-step accumulation effect, but it is also consistent with a model that is worse at short horizons and better at long horizons because of how it propagates variance.

A “general predictive advantage” claim is false. The allowed claim is horizon-specific.

Four horizons were reported. H20 was primary, but the H1 reversal is part of the result. No family-wise confirmatory claim is made. A Bonferroni threshold for four tests would be `0.0125`, not `0.05`; the reported intervals are 95% and should not be relabeled as family-wise.

---

## 6. Representation dependence

| Representation | Mean | Block-5 interval | Includes zero? |
|---|---:|---|---|
| PCA-10 energy | `+0.2644` | `[+0.0750, +0.4031]` | no |
| PCA-10 MSE | `+0.0545` | `[-0.1574, +0.2409]` | yes |
| Neuron energy | `+0.1617` | `[-0.1387, +0.4653]` | yes |
| Neuron MSE | `-0.0354` | `[-0.1682, +0.0705]` | yes |

Only the PCA-10 energy interval excludes zero. The neuron-space energy point estimate is positive but uncertain. The MSE directions are mixed.

This is why the result must be called a **fixed-PCA energy result**, not a general forecast improvement. The PCA-10 space is a training-only transform shared by both models; it is not a test-selected representation. But it is still one representation, and the result does not survive all reasonable alternatives.

---

## 7. Calibration: the energy advantage is partly spread

### 7.1 Coverage

| Model | 50% | 80% | 95% | Mean PCA-10 MSE |
|---|---:|---:|---:|---:|
| B3 | `0.546` | `0.830` | `0.938` | `1.7067` |
| B5-10 | `0.349` | `0.563` | `0.725` | `1.7612` |

B5 is severely undercovered at every nominal level. At 95%, it covers `72.5%` of observations, not `95%`. B3 covers `93.8%`, close to nominal.

The coverage is **marginal per-dimension**, averaged over the ten PCA dimensions. It is not joint 10-D coverage. Joint coverage for both models is likely lower; for B5 it would be much lower.

### 7.2 Why this matters for the energy score

The energy score is:

```text
E||X-y|| - 0.5 E||X-X'||
```

It rewards a distribution that is both close to the observation and appropriately spread. A model with too-narrow ensembles is penalized even if its mean is accurate. B5’s undercoverage can therefore produce a worse energy score without a worse conditional mean.

The MSE result confirms this:

```text
B3 mean MSE   1.7067
B5 mean MSE   1.7612
difference    +0.0545
95% interval  [-0.1574, +0.2409]
```

The mean difference is about 3.1% of B3’s MSE, and the interval includes zero. The energy difference is about 9.8% of B3’s energy, with an interval excluding zero conditional on the five recordings. The energy result is larger than the MSE result and has a different uncertainty pattern.

This does not prove that B5’s undercoverage **causes** the energy gap. It shows that the current analysis does not separate calibration from conditional-mean accuracy. A spread-matched or calibration-matched comparison is required.

### 7.3 Covariance error

B5 has a lower pooled neural covariance error (`0.516`) than B3 (`0.604`), even though B5 has worse marginal coverage. This is not contradictory: a model can approximate the covariance shape while underestimating marginal spread, or vice versa. It reinforces that the calibration story is not one-dimensional.

The correct conclusion is not “B3 is better calibrated.” It is “B3’s energy advantage coexists with better marginal coverage and worse covariance error; the score is not a clean measure of mean information.”

---

## 8. Capacity and the 51/49 decomposition

### 8.1 Parameter counts

| Session | Predictors `n` | B3 parameters `n(n+1)` | B5-10 parameters `12n+110` | Ratio |
|---:|---:|---:|---:|---:|
| 0 | 103 | 10,712 | 1,346 | 7.96 |
| 1 | 127 | 16,256 | 1,634 | 9.95 |
| 2 | 123 | 15,252 | 1,586 | 9.62 |
| 3 | 118 | 14,042 | 1,526 | 9.20 |
| 4 | 121 | 14,762 | 1,562 | 9.45 |

B3 has `7.96–9.95×` more parameters. It also has arbitrary residual cross-neuron coordinates; B5 is restricted to one own-row direction plus ten common factors.

The final audit reports that B3’s residual rows are `98.6–99.9%` outside B5’s one-dimensional residual pattern (median `99.6%`). This is expected by construction: B5 cannot represent arbitrary residual rows. It proves B3 has access to structure B5 lacks; it does not prove that structure causes the score advantage, because B3 also has far more parameters and adapts a full local operator.

### 8.2 Factor variance

The ten training-only factors explain:

| Session | Variance explained | Residual |
|---:|---:|---:|
| 0 | 0.7099 | 0.2901 |
| 1 | 0.6216 | 0.3784 |
| 2 | 0.5671 | 0.4329 |
| 3 | 0.7095 | 0.2905 |
| 4 | 0.5467 | 0.4533 |

The factors retain 54.7–71.0% of training variance. B5 is not a trivial compression; it keeps the majority of variance. But it discards 29.0–45.3% of variance by construction, and its residual operator cannot reconstruct arbitrary cross-neuron residual structure.

The final audit’s point that the evaluation PCA and factor PCA are the same transform is important: B5 is not losing evaluation-space information by projecting to ten dimensions. The restriction is in the residual dynamics, not the metric.

### 8.3 The ordered split

```text
B4m - B5   = +0.2751   (51.0%)
B5 - B3    = +0.2644   (49.0%)
B4m - B3   = +0.5395
```

This sums exactly because it is an ordered decomposition:

```text
(B4m - B5) + (B5 - B3) = B4m - B3
```

It is not a variance-component estimate, not a Shapley decomposition, and not a biological partition. Different baselines give different shares:

| Baseline | Common share | Residual share |
|---|---:|---:|
| B4m | 51.0% | 49.0% |
| B1m | 23.5% | 76.5% |
| B0 | 72.4% | 27.6% |

The 51/49 split is a useful descriptive sentence only if the baseline is named. “Common factors explain half” is not a unique statement.

### 8.4 No capacity-matched control

A capacity-matched control would need, for example:

- a restricted B3 with a rank-`k` residual operator and the same parameter budget as B5;
- a B5 with a local factor basis and the same adaptivity as B3;
- a model with the same number of parameters but no cross-neuron residual rows;
- a model with the same residual rows but a smaller parameter count.

None was prespecified. The current B6 is a coefficient ablation, not a refitted restriction. B4m/B5 are clean restrictions but not capacity-matched.

Therefore:

```text
S11: "The residual B3 advantage is information rather than estimator capacity"
status: NOT_EVALUABLE
```

This is the single most important unresolved scientific question in the project.

---

## 9. Multiplicity and selection

The analysis reports:

- four horizons;
- five sessions;
- four factor counts (1, 3, 5, 10);
- multiple representations (PCA energy, PCA MSE, neuron energy, neuron MSE);
- multiple surrogates;
- multiple bootstrap block lengths;
- multiple model baselines.

Not all of these are confirmatory tests. Many are descriptive sensitivity analyses. But the final claim is often read as if the H20 PCA-10 energy result were the only test.

The audit did not apply a formal family-wise correction because the analysis is exploratory and the tests are dependent. Instead it requires:

- the primary contrast to be named before the sensitivity analyses;
- the H1 reversal to be reported;
- the representation dependence to be reported;
- the block-length dependence to be reported;
- no “best-looking” subgroup to be promoted.

The H20 choice was prespecified in the common-state protocol. The sign rule was not independently prespecified. The test block was reused. These facts limit confirmatory weight even before multiplicity.

---

## 10. What would make the inference stronger

### Minimum for a defensible exploratory claim

1. Report both the conditional and recording-level intervals.
2. Report the effect size and the MSE null.
3. Report the H1 reversal and session 2.
4. Report the calibration difference.
5. Report the capacity ratio.
6. Stop calling the result a general forecast improvement.

### For a stronger scientific claim

1. **Independent cohort:** new animals, same protocol, animal-level split.
2. **Animal-level design:** at least 10–20 animals, with animal as the unit; no repeated test reuse.
3. **Capacity-matched control:** a restricted model with the same parameter budget and residual rank as B5.
4. **Calibration-matched comparison:** match ensemble spread or evaluate a proper score that separates calibration from resolution.
5. **Prospective preregistration:** sign, horizon, metric, baseline, and exclusion rules fixed before data access.
6. **Intervention validation:** matched stimulation/sham data for any biological claim.

### For a biological claim

None of the above is sufficient without intervention. Observational prediction cannot identify direct coupling. A positive capacity-matched result would still be compatible with an unmeasured common input, behaviour, nonlinearity or slow state.

---

## 11. Strongest defensible statistical statement

> In five Kato `WT_NoStim` recordings, B3 had lower H20 PCA-10 energy than the ten-factor control B5-10 in four of five sessions (equal-session mean `+0.26`; conditional block-5 interval `[+0.08, +0.40]`; 64% of 175 overlapping landmarks). The effect is small (`dz ≈ 0.27`), reverses at H1 (`-0.06`), is absent for mean MSE (`+0.05`, interval including zero), and is accompanied by a large coverage difference (B3 `0.938`, B5 `0.725`). A bootstrap that resamples recordings gives `[-0.04, +0.48]`, so the result is exploratory and does not establish animal-level generalization. B3 has 7.96–9.95× more parameters than B5, so the contrast does not separate residual cross-neuron information from estimator capacity.

This is the sentence the data support. Anything stronger requires new data or a new design.
