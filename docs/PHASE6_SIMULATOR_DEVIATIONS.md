# Phase 6 simulator deviations and decisions

Changes from the frozen protocol, with timing relative to test inspection.

| Decision | Timing | Reason | Consequence |
|---|---|---|---|
| Default support threshold changed to validation 99th percentile | before test | training-only LOO envelope flagged 79% of selected-model states; no external calibration | OOD fractions drop; old threshold retained for continuity |
| Real history window replaces repeated `x0` | before test | confirmed timing/history bug | GRU results change; h1 models reproduce exactly |
| Autocorrelation metric rewritten | before test | old formula correlated concatenated blocks | corrected errors are higher |
| Gate 2 uses free-running skill | before test | one-step ΔR² is not dynamical evidence | Gate 2 is now fail |
| Switching VAR dynamics restricted to diagonal | before test | full switching matrices had large transient amplification and catastrophic validation MSE | model is compact/stable but cannot test cross-neuron switching |
| Switching VAR cap applied to eigenvalues of A, not I+A | before test | the first cap was mathematically incorrect for the shift | stable but still failed skill |
| Perturbation grid reduced to q=1, durations 1/5, 3 frames, 15 rollouts | before test | full grid was computationally prohibitive on CPU | less dose coverage; support conclusion still evaluable |
| Perturbation controls use one deterministic non-candidate neuron | before test | matched random control; not a biological control | specificity is descriptive only |
| B4 not fitted | before test | trigger required ≥3/5 positive validation worms; observed 0 | no nonlinear model |
| No blend/persistence-shrinkage model added | after validation, before test | would have been selected post hoc after seeing negative validation | not used; no test-tuned model |
| H=200 stability subset limited to VAR/stochastic/switching | before test | diagnostic scope | not a full model comparison |

## Protocol deviations not made

- No test-driven hyperparameter change.
- No post-hoc gate relabelling.
- No reselection after test inspection.
- No replacement of the Randi Phase 6 reports.
- No push, tag or history rewrite.
