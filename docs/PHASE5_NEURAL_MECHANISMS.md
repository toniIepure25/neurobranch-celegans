# Phase 5D: SAAV and compact-circuit mechanism analyses

## Status and scope

**Observational encoding/decoding only.** These analyses ask whether aligned
neural activity carries direction-related information beyond head curvature and
whether that information is stable across animals. They do not establish
necessity, sufficiency or causal control. A null result is bounded by this
representation, alignment, sample size and model class.

- Events: 250 phase-aligned wild-type reversals (62 dorsal, 188 ventral), 26
  animals, from the raw Dryad reconstruction.
- Windows: 48 phase-sampled frames ending at the last annotated reversal frame.
- Behaviour baseline: six summary statistics of head curvature (mean, SD, last,
  min, max, linear trend).
- Neural feature: the same six statistics for one identified neuron, added to
  the behaviour baseline.
- Split: animal-disjoint five-fold cross-fitting; scaling on training animals
  only; class-balanced logistic regression; train-prior logit correction.
- Uncertainty: animal-cluster paired bootstrap; Holm correction within the
  nine-channel observed family.

## Incremental information over head curvature

Negative Δ log loss means the neuron adds information; positive means it hurts.
The primary metric is held-out event log loss.

| channel | n events | n animals | behaviour log loss | +channel log loss | Δ log loss [95% CI] | Δ AUROC | Holm p |
|---|---:|---:|---:|---:|---|---:|---:|
| SAAV | 250 | 26 | 0.506 | 0.516 | +0.010 [−0.008, +0.029] | −0.006 | 1.00 |
| SAAD | 75 | 9 | 0.555 | 0.567 | +0.012 [−0.072, +0.091] | −0.076 | 1.00 |
| SMDV | 61 | 6 | 0.554 | 0.423 | **−0.131 [−0.154, −0.107]** | **+0.313** | **<0.001** |
| SMDD | 151 | 15 | 0.541 | 0.486 | −0.055 [−0.144, +0.030] | +0.101 | 1.00 |
| RMDV | 38 | 4 | 0.893 | 0.974 | +0.081 [−0.025, +0.138] | +0.042 | 0.96 |
| RMDD | 49 | 6 | 0.276 | 0.294 | +0.019 [−0.005, +0.045] | −0.133 | 0.96 |
| RIV | 61 | 7 | 0.687 | 0.743 | +0.056 [−0.029, +0.140] | −0.098 | 0.96 |
| SMBV | 138 | 14 | 0.515 | 0.519 | +0.004 [−0.026, +0.034] | −0.014 | 1.00 |
| SMBD | 164 | 16 | 0.545 | 0.583 | +0.038 [+0.011, +0.071] | −0.085 | 0.032 |

Only SMDV shows a large, Holm-significant improvement. It is based on 61 events
from six animals and is not stable across animals: five animals contribute a
mean dorsal−ventral effect, with a **negative** aggregate difference
(−0.022, 95% CI [−0.119, +0.072]) and only 40% positive. The temporal-control
analysis shifts the SMDV window by 24 frames while preserving its temporal
statistics; the control improvement is much smaller (Δ log loss −0.024, 95% CI
[−0.063, +0.024]) than the observed-alignment effect. This is consistent with an
alignment-dependent, animal-specific signal, not proof of a general mechanism.

SAAV itself does not add proper-score information over head curvature. SMBD and
SAAD point estimates are adverse or unstable. RMDV/RMDD/RIV subsets are too
small for stable estimates.

## Cross-animal stability

Dorsal−ventral aligned activity, averaged over the window, per animal:

| channel | animals with both directions | mean difference [95% CI] | fraction positive |
|---|---:|---|---:|
| SAAV | 20 | −0.019 [−0.091, +0.054] | 0.40 |
| SAAD | 7 | −0.126 [−0.260, −0.016] | 0.29 |
| SMDV | 5 | −0.022 [−0.119, +0.072] | 0.40 |
| SMDD | 12 | +0.001 [−0.048, +0.054] | 0.42 |
| RMDV | 4 | +0.003 [−0.019, +0.019] | 0.75 |
| RMDD | 3 | −0.026 [−0.037, −0.010] | 0.00 |
| RIV | 6 | −0.164 [−0.296, −0.033] | 0.17 |
| SMBV | 10 | +0.015 [−0.081, +0.098] | 0.60 |
| SMBD | 13 | +0.008 [−0.087, +0.116] | 0.54 |

The aggregate signs are not a substitute for per-animal consistency. SAAD and
RIV have negative aggregate effects but only a minority of animals are positive;
RMDD has only three animals. SAAV is centred near zero with a wide interval.
These are not stable population-level direction codes in this reconstruction.

## Temporally matched controls and matched ablations

- **Observed alignment vs temporal control:** SMDV's observed Δ log loss is
  −0.131; rolling its window by 24 frames gives −0.024 with a CI crossing zero.
  For SAAV, observed +0.010 vs control +0.0075; for SMDD, observed −0.055 vs
  control −0.020; for SMBD, observed +0.038 vs control +0.027. Only SMDV shows a
  clear separation, and it remains a six-animal observation.
- **Matched feature ablation:** "drop channel" means removing the channel from
  the behaviour+channel model; it is a model ablation, not a biological
  ablation. It does not establish that the neuron is necessary.
- **Multiple testing:** the Holm family is the nine observed channels; controls
  are reported separately and are not used to rescue a failed observed test.

## Beyond-motor information

The behaviour baseline contains only head curvature summaries. "Beyond motor"
therefore means beyond this restricted head-curvature representation, not beyond
all motor variables. Velocity, angular velocity, pumping and sensory context were
not included in this offline mechanism model because the phase-aligned
reconstruction stores only head curvature and neural channels. The online
benchmark includes those variables and is the stronger test.

## What this supports

1. In this reconstruction, SAAV does not add held-out decoding information over
   head curvature at the event level.
2. The compact circuit is not uniformly informative; most channels are null or
   adverse, and the one large effect (SMDV) is small-sample and unstable.
3. There is no cross-animal stable SAAV direction code in these aligned windows.
4. A temporally misaligned control can retain some information, so alignment
   matters; but an alignment-sensitive effect is still correlational.

## What this does not support

- That SAAV is unnecessary or uninvolved.
- That SMDV causes dorsal/ventral turning.
- That any decoding result can be converted into a stimulation target.
- That the published paper's stronger SAAV result is wrong.

The most likely explanations for the difference remain protocol/preprocessing
(especially the missing aligned CSVs), event selection, alignment definition,
sample size, and the published accuracy being section-level and class-balanced
rather than animal-disjoint event-level log loss.

Machine-readable outputs: `reports/phase5/mechanisms/mechanism_channel_tests.csv`,
`mechanism_stability.csv`, `mechanism_animal_effects.csv`,
`mechanism_matched_ablations.csv`, `mechanism_predictions.parquet`,
`mechanism_summary.json`; figures in `reports/phase5/mechanisms/`.
