# Supervisor brief

## Question

Can the improved adaptive VAR forecast be turned into a calibrated
probabilistic simulator of future neural activity?

## Answer

**No, not with this model.** The conditional mean is better than the frozen VAR,
but the stochastic model built from its own innovations is badly
overdispersed. It is worse than simple probabilistic baselines on proper scores
and its 95% intervals cover 99.9% of held-out outcomes.

More importantly, the deterministic advantage is not evidence for cross-neuron
dynamics. Once a rolling local mean or an independent per-neuron AR is allowed to
adapt, the full population VAR is statistically indistinguishable from either.

## Three results

1. **The old number reproduces.** Frozen H20 MSE `1.597`, rolling w500 `0.941`,
   paired difference `0.657`, block-bootstrap CI `[0.326, 0.986]`. The `41.1%`
   and `33.9%` summaries are both correct but answer different questions.

2. **Local tracking explains the gain.** Full VAR minus local mean is `+0.026`
   `[−0.129, +0.152]`; minus independent AR is `+0.011` `[−0.080, +0.078]`.
   Both intervals include zero. The full VAR wins in three of five sessions but
   loses to the local mean in two.

3. **The stochastic extension fails.** Primary energy at H20 is `4.605`, versus
   `3.263` for local mean and `3.077` for independent AR. Coverage at
   50/80/95% is `0.904/0.992/0.999`; total standard deviation is `3.4×` the
   observed value. The block bootstrap interacts with a near-unit-root operator
   and inflates variance.

## What this means

- The adaptive VAR is a better **point predictor** than the frozen model on this
  split. It is not a demonstrated model of cross-neuron coupling.
- A good conditional mean is not a good generative model. Distributional
  calibration must be tested separately.
- The failure is structural and informative: the residual covariance, the
  operator's spectral radius and the observed variance cannot all be satisfied
  by this parameterization.
- A post-hoc diagonal-innovation variant is a useful lead: it beats the local
  mean on energy (`+0.425` `[+0.078, +0.789]`) but not reliably independent AR
  (`+0.239` `[−0.018, +0.470]`) and remains overdispersed. It is not a validated
  fix.

## Decision

- Gate A: pass.
- Gate B: fail (no added cross-neuron information).
- Gates D–F: fail (proper score, calibration, variance fidelity).
- Gate H: not evaluable; no interventional data.

No perturbation screen was run. Propagating an intervention through a model with
`3.4×` inflated variance would produce a precise-looking but indefensible
answer.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Recommended next step

Do not add complexity yet. First determine whether the variance failure is a
parameterization problem or a fundamental limitation:

1. fit state-conditioned innovation scale on training data only;
2. compare diagonal, low-rank-plus-diagonal and full covariance under the same
   mean operator;
3. evaluate on a held-out session and, if possible, a genuinely unseen
   recording;
4. require proper-score improvement **and** coverage/variance calibration before
   any trajectory or perturbation claim.

This is a negative result worth keeping. It prevents a misleading generative
story and narrows the next experiment to a testable question.

## Artifacts

- Protocol: `docs/PROBADAPT_RESEARCH_PROTOCOL.md`
- Full results: `docs/PROBADAPT_RESULTS.md`
- Decision gates: `reports/simulator/probabilistic_adaptive/prob_gates.json`
- Dashboard: “PROBABILISTIC ADAPTIVE SIMULATOR” tab.
