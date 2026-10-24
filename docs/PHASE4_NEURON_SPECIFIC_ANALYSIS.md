# Phase 4 neuron-specific analysis

## What can and cannot be claimed

The archive contains identified traces in the MATLAB structures but no
trial-level single-cell perturbation.  Everything below is a predictive or
descriptive analysis of observational activity.  A feature-ablation result is
a property of a model, not an experimental neuron ablation.  No result here
supports a claim that stimulating or silencing SAAV, RMDV, RMDD, SMDV, SMDD,
RIV, SMBV or SMBD causes a turn direction.

## Coverage

| marker | WT animals | octanol | diacetyl | tdc1 |
|---|---:|---:|---:|---:|
| SAAV | 27 | 14 | 13 | 6 |
| RMDV | 12 | 6 | 6 | 6 |
| RMDD | 16 | 7 | 9 | 5 |
| SMDV | 9 | 2 | 7 | 2 |
| SMDD | 31 | 19 | 12 | 7 |
| RIV | 12 | 7 | 5 | 5 |
| SMBV | 34 | 22 | 12 | 10 |
| SMBD | 35 | 22 | 13 | 7 |

Only 10 of 115 identified labels are present in at least 80% of WT animals.
Missing labels are zero after causal z-scoring, not imputed from another
animal.  SAAV is available in fewer than half of the WT animals, so the
primary SAAV test is necessarily a sparse-label test.

## SAAV

All WT, behaviour + SAAV versus behaviour: Δ AUROC **−0.0042**
([−0.0117, +0.0029]).  In the 27-animal SAAV-available subset:
Δ **+0.0045** ([−0.0127, +0.0210]).  SAAV-only AUROC is 0.482 in that subset.
There is no evidence that SAAV activity adds prospective turn-direction
information beyond behaviour.

This is not a refutation of the paper's SAAV manipulation.  The paper's
optogenetic evidence is aggregate and interventional; the present analysis is
observational, animal-disjoint and sparse.  A null decoding result and a
positive perturbation result can coexist.

## Circuit representation

The prespecified eight-neuron circuit model gives Δ AUROC **−0.0097**
([−0.0380, +0.0191]) across all WT.  In octanol alone it is negative with a CI
excluding zero (−0.0644, [−0.1014, −0.0322]); in diacetyl it is positive but
uncertain (+0.0379, [−0.0267, +0.1042]).  The cohort heterogeneity is not a
stable circuit effect.

## Identity permutation and ablation

Permuting the eight circuit features within each animal leaves the circuit
AUROC essentially unchanged (0.5824 vs 0.5857), consistent with the circuit
model being weak rather than identity-specific.

The prespecified and matched-random feature ablations are in
`reports/phase4/turn_direction_neuron_ablation.csv` and
`turn_direction_neuron_ablation_raw.csv`.  The full circuit model has AUROC
0.5857.  Dropping one prespecified feature changes AUROC by −0.0040 to +0.0050;
24 matched random drops have mean +0.0012 (SD 0.0033), and every prespecified
empirical p is ≥ 0.12.  No single circuit feature has an effect distinguishable
from dropping a random matched feature.  These ablations test model dependence;
they do not test whether the neuron is necessary in the living animal.

## Descriptive peak timing

The mean peak times in `turn_neuron_peak_times.csv` differ by direction for
several markers, but the standard deviations are 5–9 frames and the event
counts are unbalanced.  These are marginal descriptive statistics, not fitted
sequence onsets and not a statistical test of sequence order.  See
`PHASE4_NEURAL_BRANCH_DYNAMICS.md` for the table.

## Bottom line

The neuron-specific evidence in Phase 4 is **not supportive of a robust,
identity-resolved prospective turn-direction code** at the tested resolution.
It is also not evidence against the published interventional role of these
neurons.  The correct next step is a trial-level perturbation dataset with
animal-disjoint controls, not a stronger claim from these observational
features.
