# Supervisor brief

## What was asked

Which aspects of the recorded Kato neural population improve joint trajectory
forecasts, are the inferred interactions robust, and how much can legitimately be
learned about virtual perturbations?

## What was done

- Reproduced the frozen J0/J1/J2/J5 H20 results exactly on 175 landmarks
  (maximum difference `8.9e-16`).
- Compared B3 population VAR against B1 independent AR, B4r independent AR with
  a matched 500-frame window, B4 diagonalized operator, B2 local mean, B0
  persistence, and matched circular/block temporal surrogates, all with the same
  J5 innovations.
- Audited 103-dimensional operator spectra, non-normal transients, segment and
  bootstrap stability, and complex-mode periods.
- Fitted six training-only neural states and compared transition statistics with
  Markov and marginal nulls.
- Applied a bounded `+1 SD` mathematical displacement to six named neurons across
  ten permitted fits, with matched controls and OOD support.

## What was found

1. **Cross-neuron information is predictive.** B3 beats B1 on H20 energy by
   `+0.371` (95% session-block interval `[0.128, 0.530]`; 60.6% of landmarks
   better), and beats the matched 500-frame independent refit by `+0.547
   [0.355, 0.698]` (72% better). The same-innovation comparison is the cleanest
   evidence that shared lag structure, not a different innovation stream, drives
   the gain.
2. **The gain is not a clean mean-forecast gain over the best baseline.** B3
   versus B1 H20 mean-forecast MSE is `+0.250`, interval `[-0.029, 0.432]`. B3
   versus the weaker 500-frame independent refit improves MSE by `+0.428
   [0.191, 0.621]`. The energy advantage includes ensemble spread/shape, and
   window length is partly confounded. This must be stated, not hidden.
3. **The operator is only partly stable.** Spectral radius is stable because the
   cap binds; cross-segment off-diagonal coefficient correlation is only `0.052`.
   Non-normality is large (1.5-2.7) and transient amplification is 3.6-4.7x.
   Complex eigenvalues are common but do not match the observed slow spectra; no
   oscillation is claimed.
4. **State transitions fail the null test.** B3 transition TV is `0.053`, worse
   than the training-only Markov null at `0.031`. Occupancy is better than
   persistence, but the switching law is not validated.
5. **Virtual perturbations are not robust.** Median target response cosine across
   fits is `0.556`; only AVEL exceeds `0.80`. Some fits reverse the response
   direction. Median OOD is `1.4%`, but the maximum reaches `13.5%`.
6. **Biological validation is `NOT_EVALUABLE`.** There is no matched intervention
   recording, no sham, no dose, and no independent animal cohort.

## What cannot be said

- that a named neuron causally drives the response;
- that the fitted off-diagonal coefficient is a synapse or direct interaction;
- that a perturbation would work experimentally;
- that any neuron is a validated stimulation target;
- that the simulator is calibrated.

The synthetic audit shows why: a common input with no direct coupling produces a
spurious off-diagonal norm of `0.313` and a worse-than-diagonal forecast. The
Kato design cannot separate direct coupling from a shared drive.

## Decision

Do **not** run another architecture search on this reused split. Preserve the
positive predictive result and the negative operator/state/perturbation results.
The next credible step is an independent-animal, sham-controlled,
neuron-identified perturbation study with a pre-registered prediction.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
**Biological validation: `NOT_EVALUABLE`.**
