# Phase 3 behavioural validation

**Primary dataset:** Baskoylu & Flavell (2026), 27 freely-moving animals,
1,162 reversals, 43,200 frames.
**Question:** does recorded neural activity improve *prospective* prediction of a
real behavioural transition beyond behavioural history?
**Evaluation:** leave-one-animal-out, causal features, prespecified 6 s primary
horizon.
**Artifacts:** `reports/phase3/prospective_metrics_*.csv`,
`prospective_deltas_*.csv`, `prospective_summary.json`,
`prospective_c_sensitivity.csv`.

This is an observational prediction test. It is not an intervention and does not
identify a causal neuron.

## 1. Target and prevalence

A positive label means a reversal **onset** in the half-open future interval
\((t, t+H]\) seconds. Frames already in a reversal are ineligible, and every row
has a complete future window. Onsets come from the validated
`reversal_events` table, not from a shifted or smoothed behavioural state.

| Horizon | Frames | Eligible rows | Positive prevalence | Behaviour-only log loss | Constant log loss |
|---:|---:|---:|---:|---:|---:|
| 3 s | 5 | 29,378 | 18.94% | 0.4087 | 0.4863 |
| **6 s (primary)** | **10** | **29,293** | **35.47%** | **0.5927** | **0.6521** |
| 12 s | 20 | 29,149 | 60.80% | 0.6283 | 0.6730 |

The high prevalence is real: 31.3% of frames are inside a reversal and events are
clustered. A 6 s window is therefore not a rare-event task, and a model can look
accurate while adding little information. This is why log loss and a behavioural
baseline, not accuracy alone, are the decision metrics.

## 2. Models compared

All models see the same eligible rows and the same held-out animal. Behavioural
features are current velocity, speed, acceleration, head angle, angular velocity,
arena distance, forward-run age and causal 3/6/12 s window summaries. Neural
features are permutation-invariant summaries of causally z-scored ROI activity
at lags 0, 1, 2, 4 and 8 frames. ROI indices are **not** assumed to be identities
across animals.

The primary neural comparison is:

\[
\Delta = \text{log loss}(\text{behaviour} + \text{population neural}) -
         \text{log loss}(\text{behaviour-only})
\]

A negative value means neural activity helped. Uncertainty is a 10,000-resample
animal-level bootstrap (seed 17) of the 27 paired differences.

## 3. Primary result (6 s)

| Model | Mean log loss | Brier | Balanced accuracy | AUPRC |
|---|---:|---:|---:|---:|
| Constant training prevalence | 0.6521 | 0.2297 | 0.500 | 0.3547 |
| Behavioural state only | 0.6211 | 0.2150 | 0.569 | 0.5449 |
| Behavioural history | 0.5927 | 0.2020 | 0.617 | 0.5868 |
| Behaviour + population neural | **0.5856** | **0.1992** | 0.622 | 0.5880 |
| Behaviour + train-only PCA(10) neural | 0.5842 | 0.1986 | 0.623 | 0.5938 |
| Neural population only | 0.6495 | 0.2285 | 0.500 | 0.3991 |

**Primary incremental result:** \(\Delta = -0.00701\), 95% animal bootstrap CI
**[-0.01483, +0.00053]**, 20/27 animals improved.

The prespecified criterion was: mean Δ < 0, CI upper < 0, and at least 70% of
animals improved. The direction and animal count pass, but **the CI includes
zero, so the primary criterion is not met**. The correct conclusion is
**inconclusive/negative**, not "neural activity has no information".

The effect is small relative to between-animal heterogeneity: per-animal Δ ranges
from **-0.0630** to **+0.0389** (SD 0.0203). Three animals account for most of the
positive tail; three different animals account for most of the negative tail.
This is not a uniform biological effect. Test-set calibration is good for both
models (slope 1.005 for behaviour, 0.963 for behaviour+neural; intercepts 0.008
and -0.012), so the null is not caused by a badly scaled probability.

## 4. Secondary horizons

| Horizon | Behaviour | + Neural | Δ | 95% CI | Animals improved | Primary-criterion met? |
|---:|---:|---:|---:|---|---:|---|
| 3 s | 0.4087 | 0.4056 | -0.00306 | [-0.00740, +0.00105] | 19/27 | No |
| 6 s | 0.5927 | 0.5856 | -0.00701 | [-0.01483, +0.00053] | 20/27 | No |
| 12 s | 0.6283 | 0.6208 | -0.00742 | [-0.01566, +0.00173] | 20/27 | No |

The direction is consistently negative at all three prespecified horizons, but
none of the primary models clears the zero-CI rule. A consistent small direction
with a fragile CI is a hypothesis for a larger, preregistered replication, not a
confirmed effect.

Train-only PCA(10) is the only neural variant that clears the zero-CI rule at all
three horizons (3 s: -0.00444 [-0.00774, -0.00121], 21/27; 6 s: -0.00841
[-0.01475, -0.00220], 20/27; 12 s: -0.00914 [-0.01542, -0.00239], 21/27). Because
this model was one of several secondary variants, its CI is **not** corrected for
model selection and it cannot replace the primary test.

## 5. Negative controls and sensitivity

### Time-shifted neural control

Circularly shifting the neural features by 211 frames within each animal destroys
their alignment with the behaviour while preserving their marginal distribution
and autocorrelation. The shifted control did **not** improve:

| Horizon | Δ shifted neural | 95% CI | Animals improved |
|---:|---:|---|---:|
| 3 s | +0.00142 | [-0.00011, +0.00298] | 9/27 |
| 6 s | +0.00132 | [-0.00166, +0.00446] | 13/27 |
| 12 s | +0.00229 | [-0.00210, +0.00731] | 13/27 |

This is the expected null and argues against a trivial preprocessing artifact.
It does not rule out a shared arousal/session variable that is temporally
correlated with behaviour.

### Nonlinear behavioural-state control

A quadratic expansion of the current behavioural state is a stronger baseline
than the linear state model. Adding neural features to that baseline gave:

| Horizon | Quadratic behaviour | + Neural | Δ | 95% CI |
|---:|---:|---:|---:|---|
| 3 s | 0.3988 | 0.3970 | -0.00178 | [-0.00627, +0.00265] |
| 6 s | 0.5869 | 0.5817 | -0.00524 | [-0.01334, +0.00301] |
| 12 s | 0.6282 | 0.6225 | -0.00569 | [-0.01524, +0.00522] |

The neural increment shrinks and no longer approaches the criterion. Much of the
apparent population-neural signal is therefore compatible with nonlinear
behavioural state information that the linear baseline did not capture.

### Regularisation sensitivity (6 s)

| C | Δ | 95% CI | Animals improved |
|---:|---:|---|---:|
| 0.03 | -0.00724 | [-0.01513, +0.00024] | 19/27 |
| **0.10 (primary)** | **-0.00701** | **[-0.01462, +0.00028]** | **20/27** |
| 0.30 | -0.00737 | [-0.01497, -0.00006] | 20/27 |

The direction is stable; the CI upper bound sits on zero and crosses it at the
prespecified C. Reporting only C=0.3 would be cherry-picking.

### Removing feature families

Removing reversal-count features left Δ ≈ -0.00698 (CI crosses zero); removing
rolling history left Δ ≈ -0.00572 (CI crosses zero). No single behaviour family
explains the entire neural increment, but none of these reduced models clears the
criterion either.

## 6. Identified-neuron representation

The NeuroPAL export provides 170 labels, but only 52 labels are present in at
least 80% of animals. Using those 52 identity-aligned traces (absent labels
masked as zero after causal z-scoring) gave:

| Horizon | Behaviour + identified neurons | Δ vs behaviour | 95% CI | Animals improved |
|---:|---:|---:|---|---:|
| 3 s | 0.4070 | -0.00166 | [-0.00682, +0.00378] | 15/27 |
| 6 s | 0.5909 | -0.00172 | [-0.01054, +0.00699] | 13/27 |
| 12 s | 0.6216 | -0.00663 | [-0.02128, +0.00717] | 16/27 |

The identified-neuron representation does **not** provide a robust increment, and
at 6 s it is weaker than the anonymous population summary. This argues against a
simple "specific labelled cells carry the signal" interpretation. The identity
coverage is incomplete and the mapping is probabilistic, so this is not a clean
test of every named neuron.

## 7. Within-animal forward prediction

The separate chronological task (70/15/15 with a purge of at least H frames) is
weaker and does **not** show a neural benefit:

| Model | Mean log loss | Brier | Balanced accuracy |
|---|---:|---:|---:|
| Behaviour history | 0.7241 | 0.2266 | 0.622 |
| Behaviour + population neural | 0.7270 | 0.2323 | 0.613 |

Δ = **+0.00288** (neural worse), 9/27 animals improved. This is expected to be
noisier and is not pooled with the animal-disjoint test. It is a useful warning
that a small animal-disjoint increment does not automatically transfer to a
within-animal forward setting.

## 8. Interpretation

1. **No robust evidence** that recorded neural activity improves prospective
   reversal prediction beyond behavioural history at the prespecified 6 s
   horizon. The primary CI includes zero.
2. A small, consistent negative direction exists at 3/6/12 s, and train-only PCA
   clears the zero-CI rule as a secondary analysis. This is a lead, not a result.
3. The increment is not robust to a quadratic behavioural-state control, is
   absent for identity-aligned neurons, and reverses within-animal. The most
   parsimonious explanation is residual behavioural/arousal state, not a
   transition-specific neural code.
4. The time-shifted control is null, which rules out a trivial alignment bug but
   not a shared slowly varying session variable.
5. Transition accuracy alone would have been misleading: at 12 s, 60.8% of rows
   are positive, so a constant classifier is 60.8% "accurate" while its log loss
   is poor.

## 9. What would change the conclusion

- A larger, independent cohort with the same target and a preregistered
  animal-disjoint analysis.
- A behavioural baseline that includes the same nonlinear state information, so
  the neural increment is tested against a strong model.
- A neural representation with better cross-animal identity coverage or a
  validated alignment model, evaluated without transductive fitting.
- A perturbation that actually changes the transition, not a passive recording.
  Prediction is not control.

All virtual interventions in this repository remain:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
