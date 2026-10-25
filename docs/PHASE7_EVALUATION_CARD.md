# Phase 7 evaluation card

**Warning:** MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.

## Intended use

Rank target-level optogenetic hypotheses for a future *C. elegans* experiment and test whether active target selection is more sample-efficient than random selection on the Randi atlas. This is a computational benchmark, not a validated intervention.

## Data and population

- Randi et al. 2023 whole-brain optogenetic atlas, processed Phase 3 trial table: 237,639 rows, 3,371 events, 110 animals, 259 stimulated targets.
- Response: baseline-subtracted `response_mean`; pair estimates require at least 2 events.
- Licence/provenance: Randi OSF `e2syt`; raw recordings are not redistributed. Derived row-level artifacts follow the repository’s existing Phase 3–6 policy.

## Split and leakage controls

- Outer `unseen_targets`, 5 folds, 60/20/20 target-disjoint.
- Outer test and validation targets are excluded from the candidate pool.
- Selection log records candidate-pool hashes and selected targets.
- `audit.json` checks per fold that no test target was selected.
- State features, when enabled, are fitted only on outer-training events; post-stimulus frames are not read.
- Conformal calibration uses outer validation only.

## Endpoints

- Primary: pair Spearman at the 40% target budget, hybrid versus random.
- Secondary: target-profile Spearman normalised by split-half ceiling; 90% coverage; interval score; targets to 90% of full-atlas performance; direct-edge AUROC; anatomy-unexplained AUROC.
- Acquisition cost: animals and events.

## Known limitations

- No wet-lab closed loop and no prospective animals.
- The outer test is a held-out subset of the same atlas, not a new lab or genotype.
- The anatomy-unexplained label is an association heuristic, not evidence of an extrasynaptic mechanism.
- The discrepancy prior is a simple additive model; a poor prior can make the discrepancy score noisy.
- Batch acquisition refits once per budget point, so later batches in a budget use a stale model.
- Only two acquisition seeds and two ensemble members were used for the full grid.

## Decision rule

H1 is supported only if hybrid minus random is at least +0.02 absolute with a 95% paired block-bootstrap CI above zero at the 40% budget. Otherwise H1 is NOT SUPPORTED. Secondary findings are labelled exploratory and do not rescue a failed primary gate.

## Observed outcome

- H1: **NOT SUPPORTED**; hybrid − random = **+0.009366** (95% CI −0.000535 to +0.020489).
- H2 and H4: **NOT SUPPORTED** (discrepancy − uncertainty anatomy-unexplained AUROC **−0.002165**; hybrid − random direct-edge AUROC **−0.019844**).
- H3 reduced ablation: **NOT SUPPORTED** (+0.005375 over 3 folds).
- Calibration: **≈0.900** empirical coverage for all strategies.
- Leakage audit: no test target selected in any fold.
- Recommended use: exploratory target ranking only; do not treat the non-significant active-selection advantage as evidence for a prospective experiment.

## Non-claims

No causal graph, no causal effect estimate, no cross-dataset transfer, no foundation-model claim, no claim of being first in active connectomics.
