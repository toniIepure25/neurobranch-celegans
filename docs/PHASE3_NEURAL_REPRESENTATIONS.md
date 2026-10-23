# Phase 3 neural representations

**Question:** can a frozen, train-only neural representation improve prospective
behaviour prediction beyond behavioural history?
**Artifacts:** `prospective_summary.json`,
`prospective_metrics_6s.csv`, `prospective_deltas_6s.csv`,
`neuron_identity_coverage.csv`.

## 1. What is frozen

| Representation | Fit data | Dimensions | Applied to held-out animal |
|---|---|---:|---|
| Population summaries | none (fixed statistic) | 59 | yes |
| Train-only PCA(10) of population summaries | training animals only | 10 | frozen before scoring |
| Identified-neuron traces | label coverage across all animals; no outcome fitting | 52 | frozen before scoring |
| Behavioural history | none | 38 | yes |
| Quadratic behavioural state | none | 103 | yes |

The population summary is deliberately permutation-invariant: at each of five
lags it computes mean, SD, 10/25/50/75/90th percentiles, min, max, fraction >1,
fraction <-1 and a mean-difference term across recorded ROIs. This is a
conservative representation because ROI indices are not identities across
animals. The identified-neuron representation uses only labels present in at
least 80% of animals; absent labels are zero after causal z-scoring, never
assigned to a different cell.

Normalisation is causal (expanding mean/SD using current and past frames). The
published `gcamp/trace_array` is not used: it is z-scored over the whole
recording (see the dataset audit), which would leak future statistics.

## 2. Primary representation result (6 s)

| Representation added to behaviour | Log loss | Δ vs behaviour | 95% animal CI | Animals improved |
|---|---:|---:|---|---:|
| Population summaries | 0.5856 | -0.00701 | [-0.01483, +0.00053] | 20/27 |
| PCA(10), train-only | 0.5842 | -0.00841 | [-0.01475, -0.00220] | 20/27 |
| Identified 52 neurons | 0.5909 | -0.00172 | [-0.01054, +0.00699] | 13/27 |

The prespecified primary model (population summaries) does not clear the
zero-CI rule. PCA is a secondary representation and clears it, but it was not the
primary comparison and the CI is not corrected for choosing among
representations.

## 3. Does the representation survive stronger controls?

| Control | Δ at 6 s | 95% CI | Verdict |
|---|---:|---|---|
| Time-shifted neural features | +0.00132 | [-0.00166, +0.00446] | null, as expected |
| Behaviour + neural vs quadratic behaviour | -0.00524 | [-0.01334, +0.00301] | no robust increment |
| Identified neurons | -0.00172 | [-0.01054, +0.00699] | no increment |
| Within-animal forward | +0.00288 | n/a (27 animals, 9 improved) | worse |

The quadratic behavioural state is the key control. It lowers the behaviour-only
log loss from 0.5927 to 0.5869 and shrinks the neural increment from -0.00701 to
-0.00524, with a CI that crosses zero. A representation that only captures
nonlinear behavioural state is not evidence of a transition-specific neural code.

## 4. Representation geometry is not the test

No silhouette, reconstruction, 3-D separation or cluster purity is used as
evidence. A representation is kept only if it improves a proper prospective
score on unseen animals. The Phase 2 BunDLe-style geometry remains an in-sample
descriptive artifact; it is not reused as a Phase 3 claim.

## 5. Decision

**STOP** for the claim that a frozen neural representation robustly improves
prospective reversal prediction beyond behavioural history.
**ITERATE** only under stronger conditions:

1. a preregistered replication cohort, not the same 27 animals;
2. a behavioural baseline that already contains the nonlinear state information;
3. a cross-animal identity representation with documented coverage and
   uncertainty, not a single 52-label intersection;
4. a target and horizon fixed before analysis, with a correction for the number
   of representations tested;
5. an actual perturbation experiment before any control claim.

The small PCA result is a lead worth testing, not a licence to claim a neural
transition code. All interventions remain virtual:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
