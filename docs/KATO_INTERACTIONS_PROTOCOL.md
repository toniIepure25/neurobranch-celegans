# Kato interactions protocol

**Scope:** Kato et al. (2015) `WT_NoStim` only. No Dunn, Nichols, Baskoylu,
Randi, Kato `WT_Stim`, or any external intervention recording is used. Five
recordings are treated as five repeated sessions, not five documented animals.
The test split has been inspected in earlier phases and is reused here; every
result is exploratory and cannot become untouched confirmatory evidence.

## 1. Question and estimand

**Primary question.** Does the joint H20 forecasting advantage of the corrected
population VAR depend on time-lagged information shared between recorded neurons,
rather than on each neuron's own history, static covariance, or a population-wide
common trend?

**Secondary question.** Are fitted dynamical modes and local mathematical
sensitivities stable enough across permitted fits to support interpreted virtual
perturbations?

The estimand is **predictive dependence in the recorded panel**, not a synapse,
direct interaction, or biological causal effect.

## 2. Frozen representation and evaluation

- Train-only standardization, then a fixed 10-component PCA fitted on training
  frames only. The PCA is an evaluation representation; the dynamical models are
  fitted in the 103-neuron standardized space unless stated otherwise.
- Splits, purge gaps, 175 test landmarks, stride 10, H20 primary horizon, 100
  members and seed rule `7000 + session*10000 + landmark` are inherited from the
  frozen joint-dynamics experiment.
- J0/J1/J2/J5 are reproduced before any new comparison. The frozen artifacts are
  not overwritten.

## 3. Models and controls

| ID | Meaning |
| --- | --- |
| B0 | Persistence |
| B1 | Independent per-neuron AR, window 200 (frozen J1 definition) |
| B2 | Adaptive local mean, window 500 |
| B3 | Corrected population VAR, window 500, ridge 1.0, spectral cap 0.995 |
| B4 | B3 operator diagonalized, off-diagonal coefficients set to zero; intercept recomputed from the same transition means |
| B4r | Diagonal VAR refitted on the same 500-frame causal window (matched own-dynamics ablation) |
| B5 | B3 family fitted after column-wise temporal surrogates inside the causal window: circular shift (primary) and block permutation, block 25 (secondary) |

B4 is a mathematical ablation of a fitted operator. B4r is a separately fitted
own-neuron-only model. Neither is a biological lesion.

For B0-B5, the **same J5 innovation draws** are used at each landmark. J1's exact
frozen score uses its original independent innovation sampler and is reported
separately. This isolates conditional-mean differences while retaining the exact
J5 result.

## 4. Cross-neuron experiment

- Six surrogate draws per method per landmark; the landmark-level statistic is
  the median across draws. Surrogates are fitted only from values at or before
  the landmark and never cross a split boundary.
- Surrogate validity diagnostics: diagonal AR coefficient correlation with B3,
  and window lag-1 autocorrelation preservation.
- Paired improvements are `reference - model`; positive means the model has lower
  error. Intervals use a moving-block bootstrap **within each session**, then
  equal-session pooling, with block sizes 3, 5 and 10. The primary block size is
  5, fixed before the final run.
- Landmarks overlap; intervals are not independent-sample intervals and are not
  animal-level intervals.

## 5. Prespecified endpoints and gates

Primary endpoint: B3 versus B1 H20 energy score (10-D PCA evaluation).

Secondary endpoints:

- B3 versus B4r H20 energy;
- H20 mean-forecast MSE (secondary; energy and MSE can diverge);
- B3 versus B5 circular H20 energy;
- H20 trajectory energy, neural covariance error, autocorrelation error;
- state occupancy and transition TV;
- local sensitivity cosine and prediction/perturbation disagreement;
- OOD fraction of bounded perturbations.

Ablation thresholds:

| Gate | PASS criterion |
| --- | --- |
| G1 reproduction | maximum absolute difference from frozen per-landmark values `< 1e-6`; J2/J5 mean signed energy difference `< 0.01` |
| G2 cross-neuron information | B3 > B1 energy with block-5 95% CI low `> 0`; B3 > B4r energy with CI low `> 0`; B3 > B5 circular energy with CI low `> 0`; otherwise PARTIAL if only B1 passes, else FAIL |
| G3 operator reliability | median segment CV of spectral radius `< 0.25`, leading decay `< 0.35`, nonnormality `< 0.50`, and median pairwise off-diagonal correlation `> 0.50` |
| G4 neural-state transitions | B3 transition TV to observed `< min(Markov null, marginal null) - 0.02` |
| G5 sensitivity reliability | median target response cosine `> 0.80` and median sign agreement `> 0.80` |
| G6 empirical support | median target perturbation OOD fraction `< 0.05` |
| G7 generative validity | frozen joint calibration and joint trajectory gates both PASS |
| G8 biological validation | `NOT_EVALUABLE` in the absence of matched intervention evidence |

A technical gate PASS is not a biological validation.

### Protocol amendment recorded before the final six-draw run

The first implementation of B4r used the same full multivariate ridge fit as B4
and then diagonalized it, so B4r was numerically identical to B4. That is not a
matched own-neuron-only fit. B4r was corrected to a univariate independent AR
fit (window 500), and the B4r endpoint was changed from MSE to the primary H20
energy score. The B3-versus-B1 endpoint, the surrogate endpoint, the thresholds,
and the six-draw rule were not changed. Both energy and MSE are reported; the
MSE result is not allowed to rescue a failed energy result.

## 6. Operator analysis

- Training-only frozen VAR; three chronological training segments; five
  moving-block bootstrap fits (block 25); all 175 rolling landmark operators.
- Eigenvalues, singular values, spectral radius, non-normal transient
  amplification, eigenvector conditioning, decay timescales and complex-pair
  frequencies.
- Complex eigenvalues are not treated as oscillations unless their frequency and
  decay are reproducible and the observed training spectrum contains a
  compatible peak. No oscillatory biology is inferred from complex eigenvalues
  alone.

## 7. Neural states

- Six KMeans states fitted on projected training frames only, seed 23.
- Observed test state paths and model state paths over 100 frames.
- Occupancy, transition matrix, dwell time, recurrence and entropy; Markov and
  marginal nulls fitted from training states only.
- Kato macro labels are decoded from a subset of the same neurons. They are not
  independent behavioural measurements. No dorsal/ventral behavioural claim is
  made.

## 8. Sensitivity

- Named targets fixed before analysis: AVBL, AVBR, AVEL, AVER, RIVL, RIVR.
- Neuron mapping is verified separately per session; absent targets are not
  imputed.
- Three initial landmarks per session, selected by training-only state centroids
  (top training occupancy, nearest test landmark); the rule is written to
  `selected_sensitivity_landmarks.csv`.
- Perturbation amplitude: `+1.0` training standard deviation in one standardized
  neuron. This is a bounded mathematical displacement, not an experimentally
  achievable fluorescence dose.
- Two matched controls per target: nearest training variance, excluding all six
  targets. The rule is applied before inspecting response magnitudes.
- Ten permitted fits: rolling, frozen training, three chronological segments,
  five moving-block bootstrap fits.
- Finite differences are compared with the analytic Jacobian; linear agreement
  should be at machine precision.

## 9. Identifiability

The synthetic audit contains independent, directly coupled, common-input and
coupled-plus-common systems with known operators. It tests whether a fitted VAR
recovers direct coupling and whether common input can create spurious
off-diagonal structure and incompatible impulse responses. This is a diagnostic
of observational identifiability, not a biological causal proof.

## 10. Stop rules

- Do not add datasets, architectures, or a hyperparameter search.
- If cross-neuron value is unsupported, preserve the negative result and do not
  expand perturbation claims.
- Every perturbation output carries:
  `MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.`
- Biological validation remains `NOT_EVALUABLE`.
