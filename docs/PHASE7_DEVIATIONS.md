# Phase 7 deviations log

**Warning:** MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.

The preregistration was frozen before the full run. The following deviations are material and are reported with the results.

## D1 — Batch refit cadence

Planned: refit after every 8-target batch.  
Actual: the model is refit once per budget point; batches of up to 8 targets are selected with the model trained at the previous budget.  
Reason: the per-batch refit multiplied compute by roughly four without changing the scientific question. The batching constraint is identical for every strategy.  
Impact: acquisition scores within a budget become stale after the first batch; this can only make active policies weaker, not stronger.

## D2 — Two acquisition seeds

Planned: 3 acquisition seeds.  
Actual: 2 seeds (101, 202).  
Reason: full 5-fold × 6-strategy × 4-budget run cost.  
Impact: less stable acquisition-variance estimate; fold-level replication remains.

## D3 — Two model seeds for acquisition

Planned: 3 model seeds for acquisition.  
Actual: 2 seeds (17, 23); the full-atlas reference uses 5 seeds.  
Impact: the ensemble epistemic score is based on two members, so uncertainty acquisition is noisier. This is conservative for the active arm.

## D4 — State conditioning is an ablation, not primary

The primary H1 benchmark runs without state features. The state cache and projector are implemented and tested. A reduced H3 ablation (3 folds, 2 model seeds) found a mean state-minus-base pair Spearman of **+0.005375**, with 3/3 folds positive but below the preregistered +0.01 gate. H3 is therefore **NOT SUPPORTED**; the base CRO remains canonical.
Reason: preserve comparability with Phase 6 and avoid a second full grid.
Impact: H3 is reported as a reduced ablation, not as a primary result, and no state benefit is claimed.

## D5 — No wet-lab loop

The benchmark is pseudo-prospective. No animals were stimulated for Phase 7. The design list is a set of hypotheses for a future experiment.

## D6 — No exact OPhELIA/SBTG reproduction

OPhELIA and SBTG are cited as external positioning, not reimplemented. No claim of outperforming them is made.

## D7 — Profile reliability settings

Phase 7 uses 10 split-half draws and `min_events=2` for the test-target reliability ceiling, whereas Phase 6 used 20 draws and stricter eligibility. The ceiling is therefore estimated with less precision and on more targets. It is used as a normaliser, not as a primary endpoint. In the full run the normalised profile exceeded 1 for several strategies, which means the ceiling is underestimated for sparse test targets. The normalised profile is therefore reported with that caveat and is not used for H1 or any success claim; the raw profile Spearman is the interpretable quantity.

## D8 — No external cohort

No same-task external optogenetic cohort was available. Cross-dataset transfer is out of scope and is not claimed.
