# Supervisor brief: adaptive dynamics and stability

## Bottom line

A simple causally adaptive VAR materially improves short-horizon autonomous
forecasting over the validated frozen VAR, but it does not reproduce the neural
distribution and does not specifically rescue regime changes. The result is a
useful, bounded engineering finding, not a validated generative model.

## Strongest results

1. **Prespecified primary comparison passes.** Rolling VAR (500-frame window)
   reduces paired H20 MSE by 0.657 (33.9%; 95% block-bootstrap interval
   [0.326, 0.986]) versus frozen VAR. It wins 86% of landmarks and four of five
   sessions.
2. **The advantage spans horizons.** Adaptive models beat frozen at H1 through
   H100. At H20 the best aggregate candidate, EWMA h1000, reaches MSE 0.839 and
   skill +0.146 versus persistence.
3. **The calcium explosion has a clean cause.** The original rollout feeds
   predicted calcium back through the high-pass encoder. A corrected recursion
   is finite everywhere and passes a synthetic stable-calcium control, but its
   real H20 MSE is worse than persistence in four of five sessions.
4. **The half-split angle was over-interpreted.** Matched stationary surrogates
   also produce 52–60° angles; the real late-window excess is only 2–12°. There
   is gradual drift, not a clean regime switch.
5. **Distributional fidelity fails.** Adaptive forecasts have a standard
   deviation ratio of 0.24 and covariance error approximately equal to the
   signal covariance norm. The MSE gain is largely shrink-to-mean.

## What not to claim

- Do not claim biological nonstationarity as the cause of forecast failure.
- Do not claim the adaptive model is a digital twin or a generative mechanism.
- Do not claim session-level replication as animal-level replication.
- Do not claim the corrected calcium model is suitable for virtual
  interventions.
- Do not rank neurons or interpret impulses biologically.

Every perturbation statement must retain:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Decision status

| Gate | Status |
|---|---|
| A Nonstationarity characterized | PASS |
| B Numerical stability established | PASS |
| C Adaptive model advantage | PASS (primary, aggregate) |
| D Distributional improvement | FAIL |
| E Perturbation interpretation | MODEL-DEPENDENT ONLY |
| F Independent biological validation | NOT_EVALUABLE |

## Recommended next step

Do not add another architecture. The decisive next experiment is an
animal-disjoint, pre-registered test with randomized oxygen timing and a sham
schedule. The primary endpoint should be paired H20 MSE **and** variance/
covariance fidelity, not MSE alone. A model that lowers error by collapsing
variance should be rejected.
