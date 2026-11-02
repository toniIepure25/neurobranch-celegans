# Kato common-state controls: protocol

Status: **pre-registered before the final Kato evaluation was run**.
All choices below were fixed before inspecting the new common-factor, surrogate or
synthetic results. The Kato test block has already been reused in earlier project
phases; this is an exploratory reanalysis, not fresh confirmation.

## 1. Question

Does cross-neuron history provide useful H20 prospective predictive information
after accounting for:

1. each neuron's own history;
2. local intercept/mean activity;
3. shared population-wide activity;
4. low-dimensional common factors;
5. global temporal drift?

This is a conditional predictive-information question. It is not a test of direct
synapses, anatomical connectivity or neuronal causality.

## 2. Data and splits

- Kato et al. (2015) `WT_NoStim`, the five existing recordings only.
- Frozen temporal splits: train / 12-frame purge / validation / 12-frame purge /
  test. Test landmarks are the established 175 overlapping landmarks, stride 10,
  H20 primary (H1-H100 available for description only).
- No WT_Stim, Dunn, Nichols, Baskoylu, Randi or external intervention data.
- Normalization is fitted on training frames only; the fixed 10-D PCA is fitted
  on training frames only and is the primary evaluation representation.

### Dimension correction

The preceding interaction report described "103 predictors". That is the native
dimension of **session 0 only**. The actual per-session dimensions are:

| Session | Frames | Native neurons |
| --- | ---: | ---: |
| 0 | 3,137 | 103 |
| 1 | 3,134 | 127 |
| 2 | 3,059 | 123 |
| 3 | 3,311 | 118 |
| 4 | 3,021 | 121 |

Every model in this audit uses its session's native dimension. The previous
"103" claim is corrected here; the prior reports are preserved unchanged.

## 3. Models

All models are fitted causally on transitions ending at or before landmark `t`.
All comparisons use the same prediction landmarks, target frames and the same J5
innovations.

| ID | Definition | Window | Penalty | Role |
| --- | --- | ---: | --- | --- |
| B0 | Persistence plus common innovations | n/a | none | lower bound |
| B1h | Historical independent AR (`JD` J1) | 200 | weak effective ridge | reproduction only |
| B1m | Matched independent AR, own neuron only | 200 | ridge 1.0 on standardized design | own-history baseline |
| B2 | B1m plus a local linear time trend per neuron | 200 | ridge 1.0 | global-drift control |
| B4m | Matched independent AR, own neuron only | 500 | ridge 1.0 | matched-window own-history baseline |
| B5-k | B4m plus `k` training-fitted common factor scores | 500 | ridge 1.0 | strongest common-factor control |
| B3 | Full population VAR | 500 | ridge 1.0 | population model |
| B6 | B3 coefficients with off-diagonals zeroed | 500 | inherited | coefficient-ablation diagnostic |

### Matching assumptions

- B3 and B5-k use the same 500-frame window, ridge 1.0, unpenalized intercept,
  feature standardization and spectral cap 0.995.
- B1m and B2 use the same 200-frame window and matched ridge. B4m is the same
  own-neuron model with the 500-frame window used by B3/B5.
- B3's penalty is applied to standardized features with penalty `1.0 * I`.
  B1h's historical penalty is effectively much weaker (`1` added to a sum of
  squares over ~200 transitions). B1h is therefore not the fair own-history
  control; B1m/B4m/B5 are. The difference is reported explicitly.
- B6 inherits coefficients optimized for the full model. It is a coefficient
  ablation, not a refitted restricted model. B4m/B5 are the clean refitted
  restrictions.

### Common factors

For each session, a 10-component PCA is fitted on training frames only. Factor
scores are `f_t = P_k^T (x_t - mean_train)`; the common component is
`L f_t` with `L = P_k`, and the residual is `r_t = x_t - L f_t`.

The factor set is fixed before evaluation: `k in {1, 3, 5, 10}`. The **primary**
common-factor baseline is `B5-10`, the maximal prespecified control; `k=3` is the
parsimonious secondary. No factor count is selected on the test block.

B5's factor scores follow a fitted factor VAR inside the same causal window.
Each neuron's next value is regressed on its own current value and the current
factor scores. This is a rank-`k`-plus-diagonal approximation, not an anatomical
factor model.

## 4. Evaluation

- Primary: paired H20 energy advantage of B3 over B5-10. Energy is
  lower-is-better, so the positive advantage is defined as `B5 energy - B3
  energy`; positive means B3 is better. Moving-block bootstrap within session,
  primary block 5, 1,000 draws, equal session pooling; blocks 3 and 10 are
  sensitivity checks. This is a sign-convention clarification; the target
  quantity (whether B3 has lower energy than B5) is unchanged.
- Secondary: conditional-mean H20 MSE; joint trajectory energy; 50/80/95%
  coverage; neural covariance error; autocorrelation error; factor-subspace and
  residual-subspace energy; neuron-space energy/MSE; H1/H5/H10/H20 error growth;
  session-level effects.
- `B4m - B3` is the matched-window population advantage before common factors.
  `B4m - B5` is the common-factor increment. `B5 - B3` is the residual
  cross-neuron increment. Positive values mean the candidate has lower energy.
- Overlapping landmarks are not independent observations. No animal-level
  interval is claimed; the five sessions are repeated recordings.

## 5. Controls

### Surrogates

All shifts are restricted to the causal 500-frame window and never access a
future frame.

1. `circular_full`: circularly shift each neuron independently; preserves each
   neuron's marginal and circular autocorrelation, destroys cross-neuron
   alignment and common-factor alignment.
2. `block_full`: independent 25-frame block permutation per neuron; preserves
   coarse marginals/autocorrelation, destroys fine alignment.
3. `common_preserved_residual_shift`: preserve the observed factor trajectory
   `F` exactly, circularly shift each residual column independently, reconstruct
   `L F + R_shift`; preserves common factors and destroys residual cross-neuron
   alignment.
4. `residual_preserved_factor_shift`: preserve residual columns, circularly shift
   factor-score columns, reconstruct; preserves individual residuals but destroys
   the common-factor temporal alignment.

Six draws per surrogate per landmark. A surrogate that destroys common activity
cannot isolate residual coupling; that is why control 3 is primary for the
residual question and control 4 is diagnostic for factor dependence.

### Synthetic systems

Five scenarios, 20 neurons, 2,400 frames, three seeds (11, 23, 37), a 1,600-frame
training block and a 1,200-frame causal fitting window:

- independent diagonal dynamics;
- shared external factor with no direct coupling;
- direct lagged ring coupling without a shared factor;
- both shared factor and direct coupling;
- time-varying common-factor loadings.

The synthetic analysis is an implementation diagnostic, not biological evidence.
Because 20 neurons are used, the primary synthetic common-factor control is
`B5-3`; `B5-10` is reported as a saturated diagnostic. The Kato primary control
remains `B5-10`.

## 6. Decision gates (declared before final evaluation)

| Gate | PASS criterion | Otherwise |
| --- | --- | --- |
| G1 historical reproduction | B1h/B3/B4r/J2/J5 H20 energy reproduces with max absolute difference `< 1e-6`; H20 MSE difference reported separately; J2/J5 mean difference `< 0.01` | FAIL |
| G2 fair matched baselines | all primary models share landmarks, targets and J5 innovations; B3/B4m/B5 use 500 frames and matched ridge; comparison table complete | PARTIAL if one mismatch, FAIL if more |
| G3 common-factor explanation | all five sessions and all `k` values reported; factor variance and B3/B4m/B5 decomposition computed | PARTIAL if incomplete, FAIL if not computable |
| G4 residual cross-neuron information | `B5-10 - B3` H20 energy mean `> 0`, block-5 95% CI low `> 0`, fraction better `> 0.5` | PARTIAL if positive point estimate but CI includes 0; FAIL otherwise |
| G5 synthetic specificity | direct-coupling scenarios show B3 lower than B5 in at least 2/3 seeds; common-only does not show a positive B3 advantage | PARTIAL if direct only or common-only unclear; FAIL if common-only mimics coupling |
| G6 robustness | B3 lower energy than B5 in at least 4/5 sessions and the same direction in PCA-10 and neuron space | PARTIAL if one condition; FAIL otherwise |
| G7 structural identifiability | must not pass from observational prediction; requires intervention/independent common-input measurement | fixed `FAIL` for structural claims |
| G8 biological intervention validity | requires matched sham-controlled perturbation | `NOT_EVALUABLE` |

A predictive PASS cannot pass G7 or G8. A negative G4 is a legitimate final
result and will be preserved.

## 7. Seeds and reproducibility

- Historical J5 seeds: `7000 + session*10000 + landmark`.
- Common-factor PCA: deterministic full SVD, seed 0 where randomized solver is not
  used.
- Surrogate seeds: `91000 + session*1000 + landmark*11 + draw`.
- Bootstrap seed: `1700 + block`.
- Synthetic seeds: 11, 23, 37.

Outputs are written to `reports/simulator/kato_common_state/` with a SHA256
manifest. The frozen `reports/simulator/joint_dynamics/` and
`reports/simulator/supervisor_handoff/` artifacts are not overwritten.

## 8. Claim boundary

Allowed if G4 passes: "residual conditional predictive information beyond
own-neuron history and the prespecified common factors." Not allowed: direct
coupling, synapse, command neuron, causal driver, or biological stimulation
target. Biological intervention validation remains `NOT_EVALUABLE`.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
