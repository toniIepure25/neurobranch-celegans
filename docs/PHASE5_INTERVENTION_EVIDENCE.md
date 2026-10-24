# Phase 5E: intervention evidence map

## Bottom line

The public archive contains **no trial-level optogenetic or acute
cell-specific stimulation table**. It contains observational calcium recordings,
including a `tdc-1` mutant dataset, but no per-trial manipulation labels that
would license a causal effect estimate. We therefore did not create pseudo-trials
from group averages, did not treat WT versus `tdc-1` as acute SAAV stimulation,
and did not run virtual intervention optimisation.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

No model passed the Phase 5 validation gate for virtual intervention
optimisation. No simulated perturbation in this repository is a biological
intervention.

## Evidence categories

| category | what is in it | allowed inference |
|---|---|---|
| Published finding | group-level manipulations reported in Kramer et al. | The paper reports an effect; we did not re-estimate trial-level effects. |
| Available trial-level observation | none for optogenetics/silencing in Dryad | No trial-level causal estimate. |
| Simulator prediction | none run | Nothing causal. |
| Hypothesis | proposed roles of SAAV/circuit | Testable, not demonstrated. |

## Map

| id | target | manipulation | evidence type | trial-level available | causal reading |
|---|---|---|---|---|---|
| P1 | RIM | optogenetic activation | published group-level | no | reported manipulation effect; not SAAV-specific |
| P2 | RIM | optogenetic inhibition | published group-level | no | reported manipulation effect; not SAAV-specific |
| P3 | AIB | genetic silencing | published group-level | no | reported manipulation effect; not acute SAAV |
| P4 | `tdc-1` / tyramine | mutant background | published + partial public observational data | no | genotype/distribution shift, not acute cell-specific |
| P5 | SAAV | none in public archive | hypothesis | no | untested causal hypothesis |
| P6 | compact circuit | none in public archive | observational decoding | no | encoding only; model ablations are not biological |
| P7 | any candidate | virtual perturbation | simulator prediction | not applicable | not run; no validated model |

The machine-readable table is
`reports/phase5/interventions/intervention_evidence.csv`.

## Why the published manipulations do not validate our decoder

1. **Different level of evidence.** A group-level optogenetic effect on reversal
   statistics does not imply that a decoder trained on observational calcium
   can predict or control turn direction.
2. **Different target.** RIM/AIB manipulations are not SAAV-specific. They
   cannot isolate SAAV's contribution.
3. **Genotype is not acute stimulation.** `tdc-1` changes development,
   neuromodulation and behaviour distributions; WT versus mutant is not a
   cell-specific acute perturbation.
4. **No trial table.** Without per-trial stimulation, trial identity, genotype
   and outcome, any "intervention effect" computed from group means would be a
   pseudo-trial and is prohibited here.

## What would be needed for a causal claim

- cell-specific optogenetic activation/inhibition of SAAV (and matched control
  neurons) with per-trial timing and behaviour;
- sham/light-only and genetic controls;
- pre-registered direction, turn-size and reversal-length outcomes;
- animal-level randomization and blinding;
- a model validated on held-out animals and perturbations before any virtual
  optimisation.

Until then, the correct claim is observational: SAAV activity is correlated
with turn-related variables in the published data, but this project does not
show that SAAV causes the turn, and the independent reconstruction did not find
a stable incremental SAAV decoding signal.
