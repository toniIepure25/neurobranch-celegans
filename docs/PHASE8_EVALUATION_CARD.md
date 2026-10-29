# Phase 8 evaluation card

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

| Item | Frozen choice |
|---|---|
| Dataset | Randi et al. 2023 Phase 3 trial table, 3,371 events / 110 animals |
| Experimental unit | animal × support-set episode |
| Primary split | `unseen_animals`, seed **41** |
| Support/test | target-disjoint and event-disjoint within each held-out animal |
| Primary metric | pair-level Spearman correlation |
| Scalar baselines | static CRO, global mean |
| Trajectory baseline | per-responder training-mean template |
| Uncertainty | 1,000-replicate animal-cluster bootstrap (reported recovery used 100; see deviations) |
| Support budgets | 0, 2, 4, 8, 16; primary k=8 |
| Acquisition | random, uncertainty, diversity, `eig` proxy, hybrid |
| Seeds | model 17/23/31; acquisition 101/202 |
| Calibration | 90% intervals; scale fitted on validation animals only |
| H1 gate | Δ pair Spearman ≥ +0.010, CI low > 0, trajectory Δ ≥ +0.050 |
| H2 gate | k=8 vs k=0 Δ ≥ +0.010, CI low > 0 |
| H3 gate | EIG vs random Δ ≥ +0.010, CI low > 0; confirmatory only if H2 supported |
| H4 gate | trajectory > persistence +0.050; coverage 0.80–0.95 |
| Leakage gate | zero animal overlap, zero support/test target or event overlap |

## Read this before quoting a number

- A positive result is a **model-dependent prediction**, not a causal effect.
- A null H2/H3 is a bounded negative result about this atlas and this
  personalisation design; it is not evidence of absence.
- The `eig` arm uses ensemble disagreement, not a formal EIG estimator.
- H4 is a pseudo-prospective event-order rollout with observed previous
  outcomes, not a closed-loop experiment.
- Do not transfer these numbers to another species, stimulation protocol, or
  dataset without a new evaluation.
