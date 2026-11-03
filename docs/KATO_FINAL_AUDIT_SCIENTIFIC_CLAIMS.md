# Kato final audit: scientific claims

## 1. Five propositions, five verdicts

| # | Proposition | Verdict |
| ---: | --- | --- |
| 1 | Full population history improves an H20 probabilistic score. | `SUPPORTED_EXPLORATORY` |
| 2 | Population history contains predictive information beyond the specified ten-factor common-state baseline. | `SUPPORTED_EXPLORATORY`, with capacity and sign caveats |
| 3 | There are stable pairwise dynamical interactions between neurons. | `UNSUPPORTED` / effectively invalidated as a stability claim |
| 4 | The learned operator represents biological neural connectivity. | `UNSUPPORTED` |
| 5 | Mathematical perturbation response predicts actual stimulation outcome. | `UNSUPPORTED` and `NOT_EVALUABLE` biologically |

Only propositions 1 and 2 have positive evidence in the current reports.

### Proposition 1

B3 has lower H20 PCA-10 energy than B1m (`+0.3458`, interval `[+0.1018,
+0.5013]`) and B4m (`+0.5395`, interval `[+0.3470, +0.6863]`). Direction is
positive in four of five sessions for B3 vs B5, and the pooled block intervals
exclude zero. This is exploratory because the test tails have been reused and the
sessions are repeated recordings, not independent animals.

### Proposition 2

The B3-over-B5-10 residual contrast is `+0.2644` energy units, block-5 interval
`[+0.0750, +0.4031]`, 64.0% of landmarks. The ten-factor control uses the same
training-only PCA transform and the same history window, penalty, innovations and
initialization. However, B3 has 7.96–9.95× more parameters and access to all
other-neuron residual coordinates; the contrast therefore does **not** isolate
neuron-specific interactions.

The sign rule was clarified during analysis and has no independent pre-evaluation
commit. The numerical pass is retained; the evidential label is exploratory, not
confirmatory.

### Proposition 3

The fitted off-diagonal operator is unstable across chronological training
segments: median pairwise coefficient correlation `0.052`, below the `0.50`
reliability gate. Spectral-radius stability does not rescue coefficient identity.
A single fitted operator is not a stable interaction graph.

### Proposition 4

No observational prediction identifies synapses, gap junctions, direct
interactions or a command neuron. The Kato panel is a selected set of 103–127
predictors, not the connectome; calcium imaging is slow and indirect; common
input, nonlinearity and unobserved state remain compatible with the predictive
result. `G7_structural_identifiability = FAIL`.

### Proposition 5

Virtual perturbations are mathematical displacements inside a fitted linear
model. Cross-fit response cosine is only `0.556` median; some fits reverse sign.
There is no matched sham-controlled stimulation dataset, so biological
intervention validity is `NOT_EVALUABLE`. No neuron is ranked as a target and no
perturbation trajectory is a biological prediction.

## 2. Handoff wording audit

Reviewed: `docs/HANDOFF_*.md`, `docs/KATO_COMMON_STATE_*.md`,
`docs/KATO_INTERACTIONS_*.md`, and the common-state section of
`app/handoff_demo.py`.

No audited common-state or interaction text claims that the fitted operator is a
biological connectome, that a virtual perturbation is a validated stimulation
prediction, or that the ten-factor control is a measured common input. The
standing warning is present in the interaction and common-state sections.

Two representational issues remain:

1. The older executive summary names the population-versus-independent-AR result
   as the strongest positive result and predates the B5-10 control. The revised
   brief supersedes this wording; the historical file is preserved unchanged.
2. The dashboard's G4 metric labels the numerical gate without the sign-provenance
   caveat. This audit adds the caveat in the revised brief and results document;
   it does not change the displayed score.

These are versioning/clarity issues, not evidence of an accidental causal claim.

## 3. Standing labels

Every relevant handoff and demo statement must retain:

```text
PROBABILISTIC CALIBRATION NOT ESTABLISHED
MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.
Biological intervention validation: NOT_EVALUABLE.
```

## 4. Strongest allowed sentences

> In five repeated Kato WT_NoStim recordings, a full population VAR shows a
> modest, horizon-dependent exploratory H20 energy advantage over a matched
> own-history-plus-ten-factor control, but the effect is not uniform across
> sessions or representations and does not identify coupling.

> The result is consistent with residual cross-neuron predictive information,
> unmeasured common input, nonlinearity or estimator capacity; the current design
> cannot distinguish these explanations.

## 5. Forbidden sentences

- “The model recovers the functional connectome.”
- “B3 identifies direct neuron-to-neuron coupling.”
- “The ten-factor control proves that 49% of the signal is synaptic.”
- “The virtual perturbation predicts the effect of stimulating AVEL.”
- “G4 is a fully prespecified confirmatory PASS.”
- “Five animals independently replicate the effect.”
