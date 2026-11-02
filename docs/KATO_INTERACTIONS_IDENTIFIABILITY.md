# Observational identifiability

A predictive cross-neuron advantage does not identify a direct interaction. The
synthetic audit asks what a lag-one VAR can recover when the ground truth is
known. It uses eight neurons, 2,400 frames, three seeds (11, 23, 37), and four
scenarios:

- `independent`: diagonal dynamics;
- `coupled`: direct ring coupling, no common input;
- `common`: no direct coupling, one latent common input;
- `coupled_common`: both.

All fits use the same ridge, spectral cap and evaluation style as the Kato audit.

## Results (mean across seeds)

| Scenario | Off-diagonal recovery correlation | Fitted off-diagonal norm | Full/diagonal H20 MSE ratio | Impulse cosine |
| --- | ---: | ---: | ---: | ---: |
| independent | n/a (zero truth) | 0.083 | 1.000 | 0.772 |
| coupled | 0.971 | 0.397 | 0.995 | 0.992 |
| common | n/a (zero truth) | 0.313 | 1.556 | 0.399 |
| coupled_common | 0.379 | 0.941 | 1.248 | 0.321 (SD 1.04) |

## What this establishes

1. **Direct coupling is recoverable when it is the only structure.** The coupled
   scenario recovers the off-diagonal pattern at `r = 0.971` and reproduces the
   impulse response at cosine `0.992`.
2. **Common input creates spurious cross-neuron structure.** With no direct
   coupling, the common-input scenario still has an off-diagonal norm of `0.313`,
   and the full model forecasts **worse** than the diagonal model (MSE ratio
   `1.556`). The impulse response is only weakly aligned with the truth
   (`cosine 0.399`).
3. **Common input plus direct coupling is not identifiable from this design.**
   Recovery correlation drops to `0.379`; the fitted off-diagonal norm is more
   than twice the true norm (`0.941` versus `0.379`); impulse-direction agreement
   is unstable across seeds (mean `0.321`, SD `1.04`, including a negative case).
4. **A good energy score can coexist with a wrong operator.** The Kato B3 model
   improves held-out energy while its off-diagonal coefficients are unstable
   across training segments. The synthetic audit shows why an operator should not
   be read as a connectome.

## Consequence for Kato

The Kato recordings are observational, with no measured common input and no
intervention. The audit can reject a purely independent model as a complete
description of predictive dependence, but it cannot decide whether the fitted
cross-neuron coefficients are direct coupling, a shared latent drive, a
behavioural state, or an artefact of nonstationarity.

This is an **observational equivalence** problem, not a missing significance
test. More flexible models or more hyperparameter search would not solve it.
Matched perturbation data or an independently measured common input would.

## Boundary

- The synthetic system is linear and Gaussian; Kato calcium dynamics are not.
- Three seeds are a diagnostic, not a calibrated distribution over networks.
- The audit tests one lag; longer lags and unobserved inputs can change the
  equivalence class.
- The result does not say the Kato network has no interactions; it says this
  recording design cannot identify them as direct.
