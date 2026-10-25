# Phase 7 supervisor brief

**Warning:** MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.

## Question

Phase 6 tested whether a response operator generalises to unseen targets at a fixed atlas. Phase 7 asks a different question: can an active policy choose which target to stimulate next and reach the same held-out-target performance with fewer targets?

## What was built

- A pseudo-prospective, target-disjoint active-acquisition benchmark with frozen outer test targets.
- Six policies: random, uncertainty, effect, diversity, anatomy-discrepancy and a preregistered hybrid.
- A validation-only conformal calibration path and interval-score evaluation.
- An anatomy-unexplained readout, explicitly labelled as association rather than mechanism.
- An experiment designer that ranks target hypotheses and carries the non-causal warning.
- A twelfth dashboard tab, unit/integration tests, preregistration, deviations log and evaluation card.

## Why this is a real step beyond Phase 6

Phase 6 produced a static operator and a negative connectivity result. Phase 7 changes the scientific object to a measurement policy and evaluates it under the same leakage-safe target holdout. The key differentiator is not “active learning” in general; it is target-level acquisition with an anatomy-discrepancy objective and a measurement-error-aware endpoint.

## What the result can and cannot mean

A positive H1 would support sample efficiency on this atlas and justify a prospective experiment. A negative H1 bounds the value of active probing here and is reported as such. Neither result establishes causality or a wet-lab closed loop.

## Outcome (canonical run)

- H1: **NOT SUPPORTED**. Hybrid minus random at 40% = **+0.009366** (95% block-bootstrap CI **−0.000535 to +0.020489**; 6/10 paired blocks positive), below the +0.02 gate.
- Point estimates at 40%: uncertainty **0.0851**, diversity **0.0847**, hybrid **0.0839**, effect **0.0821**, discrepancy **0.0763**, random **0.0745** pair Spearman. The direction favours active selection, but the preregistered effect and CI gate are not met.
- Full-atlas reference: **0.1007** pair Spearman at the 60% budget and five seeds.
- Calibration: all strategies achieved **≈0.900** empirical 90% coverage after validation-only conformal calibration.
- H2 anatomy-unexplained AUROC (discrepancy minus uncertainty): **−0.002165**, NOT SUPPORTED.
- H4 direct-edge AUROC (hybrid minus random): **−0.019844**, NOT SUPPORTED.
- H3 state conditioning (3-fold reduced ablation): **+0.005375**, 3/3 folds positive but below +0.01, NOT SUPPORTED.
- Leakage audit: `fold_leaks = []`, `no_test_target_selected = true`; 258 unique targets selected across folds.

## Decision requested

Review the preregistered gate, the calibration result and the negative/positive decision. The canonical decision is **negative**: keep the Phase 6 operator as the canonical baseline, do not promote active probing on this atlas, and do not launch a prospective pilot on the strength of the non-significant point estimate. The top target list is an exploratory hypothesis list only.
