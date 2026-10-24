# Phase 5 next experiments

## Highest-value next experiment

**Obtain the four author-local aligned CSVs and the preprocessing script, then
run the exact official section-level decoder and a prespecified event-level,
animal-clustered re-analysis on the same arrays.**

Why this first:

- It is the only way to distinguish "the published signal is section-level and
  does not survive event/animal aggregation" from "our reconstruction lost the
  signal during alignment".
- It directly tests the leading explanation without new wet-lab work.
- It is cheap: the official code already exists; the missing inputs are the
  blocker.

Pre-registered comparison:

1. Reproduce the official 5×4 section accuracy within a stated tolerance.
2. On the same events, compute event-level AUROC and log loss with
   animal-cluster bootstrap.
3. Compare SAAV+HC vs HC at the event level, animal-disjoint.
4. If the section result reproduces but the event result is null, the
   discrepancy is an evaluation-unit/overlapping-section effect. If neither
   reproduces, the difference is preprocessing/model, and the authors' script is
   required.

## Realistic 6-ECTS package

One ECTS is roughly 25–30 hours; 6 ECTS is 150–180 hours. This is a scoped
research mini-project, not a full paper.

| work package | effort | deliverable | go/no-go |
|---|---:|---|---|
| WP0. Data request, provenance, exact rerun | 1.0 ECTS | received inputs, signed manifest, exact section-accuracy reproduction | inputs received |
| WP1. Event-level and animal-clustered re-analysis | 1.0 ECTS | paired section-vs-event analysis with CIs | exact or documented near-exact run |
| WP2. Causal-onset online benchmark on the full cohort | 1.5 ECTS | full-cohort online curves, risk sets, calibration | enough events per landmark/animal |
| WP3. Cell-resolved alignment and cross-animal stability | 1.0 ECTS | SAAV/SAAD/SMD/RMD/RIV/SMB stability with leave-one-animal-out | identifiable cells in enough animals |
| WP4. Temporal controls, matched ablations, sensitivity | 0.75 ECTS | control-adjusted effect sizes, multiplicity correction | effects survive controls |
| WP5. Write-up and reproducible package | 0.75 ECTS | report, code, tests, dashboard, data/code availability statement | all artifacts reproducible |

### Out of scope for 6 ECTS

- New optogenetics or calcium imaging.
- A causal circuit-mapping claim.
- Virtual intervention optimisation before external validation.
- A new whole-brain model.

## WP2 detail: online benchmark

- Use the full wild-type cohort, not only the 27-animal SAAV subset; stratify by
  SAAV availability.
- Compare three onsets: archival state edge, velocity threshold with confirmation,
  and a blinded human/automated annotation. Report disagreement and risk sets.
- Landmarks 0.6, 1.2, 2.4, 3.6, 4.8, 7.2 s, with a survival/risk-set curve.
- Primary metric: event log loss vs prior and vs behaviour; secondary AUROC,
  Brier, calibration slope/intercept, lead time.
- Required baselines: prior, behaviour, sensory, SAAV, compact circuit,
  population; linear and small RNN.
- Animal-disjoint outer folds; inner validation only for early stopping.
- Report per-animal effects and a leave-one-animal-out sensitivity.

Success criterion (pre-registered): a neural mode must beat behaviour on
animal-clustered event log loss with a 95% CI excluding zero, and the effect must
survive temporal-shift and missing-identity controls. AUROC alone is insufficient.

## WP3 detail: cell-resolved mechanism

- Align to the same phase and event set for every cell; no per-cell event
  selection that changes the cohort.
- Require at least 8 animals with both directions for a confirmatory test;
  otherwise label exploratory.
- Cross-animal stability: sign agreement, random-effects estimate, leave-one-out.
- Beyond-motor: add velocity, angular velocity, pumping and sensory context, not
  only head-curvature summaries.
- Temporal controls: circular shifts that preserve autocorrelation; report the
  control distribution, not one shift.
- Multiplicity: Holm across the prespecified cell family; controls separate.

## WP4 detail: causal intervention (collaboration, not 6 ECTS)

Only if a wet-lab collaborator is available:

- cell-specific SAAV activation/inhibition with per-trial timing;
- light-only and genetic controls;
- randomized trial order, blinded scoring;
- preregistered outcomes: turn direction, turn size, reversal duration;
- sham and non-SAAV neuron controls.

No pseudo-trials from group averages. WT vs `tdc-1` is not acute SAAV
stimulation.

## Decision tree

```text
CSVs received?
├─ no  -> exact reproduction remains BLOCKED; publish reconstruction + request
└─ yes -> official section accuracy reproduces?
          ├─ no  -> preprocessing/model mismatch; debug with authors
          └─ yes -> event-level animal-clustered SAAV increment?
                    ├─ yes -> mechanism follow-up (WP3), then intervention (WP4)
                    └─ no  -> evaluation-unit explanation; report section-vs-event
```

## Deliverables for the 6-ECTS package

- Reproducible environment and manifest.
- Exact or near-exact reference reproduction table.
- Event-level and online benchmark with per-animal predictions.
- Mechanism/stability analysis with controls.
- Short report and supervisor brief.
- Data/code availability statement and the unsent author request.

## What would change the conclusion

A positive, properly scored, animal-disjoint SAAV increment that survives
temporal controls and is stable across animals. To date, no such result exists
in this reconstruction.
