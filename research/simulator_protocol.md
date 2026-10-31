# Frozen simulator protocol

**Frozen:** 2026-10-31, before fitting or test evaluation.
**Full protocol:** [`docs/SIMULATOR_RESEARCH_PROTOCOL.md`](../docs/SIMULATOR_RESEARCH_PROTOCOL.md)
**Base commit:** `e18139b15048b31bc384cab1cb4591b09b80ab34`

This file records the pre-specified gates in a compact, machine-readable
form. The full document is authoritative for definitions and caveats.

| Gate | Pass condition | Failure label |
|---|---|---|
| G1 | Five worms, 20-step free-running rollouts, no future access, ≥95% finite, exact `x0`, reproducible seeds, saved manifest | no simulator claim |
| G2 | Validation-selected learned model: `ΔR²₁ ≥ 0.02` vs persistence, paired block-bootstrap CI excludes 0, `MSE₅ ≤ persistence`, in ≥4/5 worms; distribution error not >10% worse | `no_dynamical_advantage` |
| G3 | Simulated REV outcome distribution closer to observed kNN local distribution than persistence and marginal null by ≥0.02 TV, CI excludes 0 in ≥3/5 worms, entropy inside observed 95% interval | `branching_not_supported` |
| G4 | Six named targets map in all worms; three perturbation protocols distinct; ≥100 paired reproducible rollouts; OOD reported; empirical path refuses without evidence | no perturbation interface claim |
| G5 | Candidate sign stable across ≥4/5 worms, ≥2/3 seeds, ≥2/3 amplitudes/timings; above 95th percentile of matched controls in ≥4/5; max-stat corrected | `model_dependent_only` |
| G6 | Matched held-out actual interventions in the same preparation, sham controls, ≥3 animals, independent behaviour | `not_evaluable` for Kato |

Primary split: `configs/experiments/blocked_suite.yaml` (test block 2,
validation block 4). Stress split: `configs/experiments/forward_stress.yaml`.

Seeds: `17, 23, 31`. Bootstrap: moving block length 10, 2,000 replicates.

Model selection is on validation only; persistence is in the candidate set.
Test block 2 is not used for selection or threshold changes.

All unvalidated perturbation output must carry:

> MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.

Deviations are recorded in section 10 of the full protocol before the affected
analysis.
