# Phase 4 neural dynamics around the turn decision

## Purpose

These analyses describe when behaviour and neural population activity become
predictive around the reversal offset.  They are descriptive and
model-dependent.  Negative relative frames are prospective; frame 0 and later
are retrospective positive controls and are never used to claim prospective
information.

## Time-resolved decoding

A single-frame logistic model was fit leave-one-animal-out at each relative
frame from −12 to +12, using the same 945 events.

| relative frame | behaviour AUROC | neural population AUROC | behaviour + population AUROC |
|---:|---:|---:|---:|
| −12 | 0.578 | 0.541 | 0.591 |
| −6 | 0.489 | 0.552 | 0.541 |
| −4 | 0.616 | 0.544 | 0.616 |
| −2 | 0.508 | 0.561 | 0.556 |
| −1 | **0.737** | 0.539 | 0.738 |
| 0 | **0.978** | 0.509 | 0.978 |
| +2 | 0.886 | 0.522 | 0.885 |
| +6 | 0.538 | 0.450 | 0.533 |
| +12 | 0.544 | 0.495 | 0.538 |

Two facts dominate:

1. **Behaviour is strongly predictive at −1 and 0.**  The offset-adjacent head
   angle nearly determines the post-offset D/V sign.  This is expected
   autocorrelation, not a neural code.  It is exactly why the primary
   four-frame window ends at least two frames before the offset.
2. **Once the outcome-adjacent frames are excluded, behaviour drops to
   AUROC ≈ 0.49–0.62 and neural activity does not add a stable increment.**
   The neural population alone is around 0.54–0.56 at most prospective frames.

The single-frame curve is not directly comparable to the four-frame primary
model, but it shows that the apparent predictability is concentrated in the
last pre-offset behaviour sample.  It also shows why an unguarded “prediction”
at frame −1 would be misleading.

## Population-state trajectory

PCA on the eight population statistics per frame (mean, SD, quantiles,
threshold fractions, mean absolute activity) explains 60.7%, 32.7% and 4.1%
of variance in the first three components.  Trajectories are averaged within
animal first, then across animals, so animals with many events do not
dominate.

The dorsal and ventral mean trajectories separate in PC1/PC2 around the
offset.  This is a descriptive difference in population state, not evidence
that the state causes the turn.  The trajectory figure is
`phase4_turn_pca_trajectories.png`.

## Descriptive peak timing

For the eight prespecified circuit neurons, the mean relative frame of peak
causal z activity within [−12, +12] is:

| neuron | dorsal mean peak | ventral mean peak | dorsal n | ventral n |
|---|---:|---:|---:|---:|
| SAAV | −0.77 | −2.49 | 99 | 346 |
| RMDV | +0.87 | −0.98 | 54 | 129 |
| RMDD | −0.72 | −0.29 | 53 | 214 |
| SMDV | −2.06 | +3.70 | 31 | 93 |
| SMDD | +2.21 | −0.62 | 99 | 443 |
| RIV | −0.56 | +2.92 | 45 | 173 |
| SMBV | +0.55 | +1.70 | 100 | 501 |
| SMBD | +0.35 | −0.03 | 118 | 479 |

The ordering differs by direction (for example, SMDD peaks later for dorsal
turns, while SAAV and SMDV peak earlier for ventral turns).  Standard
deviations are 5–9 frames, the events are unbalanced, and these are marginal
peak times rather than fitted sequence onsets.  The table is
hypothesis-generating, not a statistical demonstration of a neural sequence.

## Controls

| control | result |
|---|---|
| neural features shifted 50 frames earlier | population Δ AUROC **−0.0164**; SAAV +0.0061; circuit −0.0011 on 913 common events |
| circuit identity permutation within animal | circuit AUROC 0.5824 vs 0.5857 original (Δ −0.0033) |
| within-animal label permutation (6 draws) | observed Δ +0.0125; null mean −0.0005, SD 0.0226; one-sided empirical p = 0.333 |
| circuit-feature ablation (24 matched random draws) | full AUROC 0.5857; prespecified drops −0.0040 to +0.0050 vs random mean +0.0012 (SD 0.0033); all p ≥ 0.12 |

The label-permutation test is deliberately small (6 draws) and is reported as
a diagnostic, not a definitive randomisation p value.  The observed increment
is well within the permutation null.  The shifted-neural control is worse than
the true population window, and identity permutation barely changes the weak
circuit model.  Neither supports a specific identity-resolved neural code.

## Interpretation

The dynamics are compatible with a system in which the upcoming turn is
already reflected in the head/body state immediately before the offset, while
the recorded population activity adds little information at the four-frame,
animal-disjoint resolution tested here.  This does not show that the neurons
are uninvolved; it shows that the current observables and model do not
decode a prospective turn direction from them.

## Outputs

- `reports/phase4/turn_dynamics_timeresolved.csv`
- `reports/phase4/turn_pca_trajectories.csv`
- `reports/phase4/turn_pca_variance.json`
- `reports/phase4/turn_neuron_peak_events.csv`
- `reports/phase4/turn_neuron_peak_times.csv`
- `reports/phase4/figures/phase4_turn_dynamics_timeresolved.png`
- `reports/phase4/figures/phase4_turn_pca_trajectories.png`
- `reports/phase4/figures/phase4_turn_neuron_peak_times.png`
