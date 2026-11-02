# Results

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
**Biological validation: `NOT_EVALUABLE`.**

## Bottom line

The Kato `WT_NoStim` panel contains **predictive cross-neuron information**: a
full lag-one population operator (B3) forecasts H20 joint trajectories better than
an independent per-neuron model (B1) when both use the same innovations. The
advantage is an energy-score advantage (`+0.371`, 95% interval `[0.128, 0.530]`),
not a conclusive mean-MSE advantage (`+0.250`, interval `[-0.029, 0.432]`).

The result does **not** identify interactions. The fitted off-diagonal operator is
unstable across training segments, the state-transition law is worse than a
training-only Markov null, and virtual perturbations of named neurons disagree
across permitted fits. The correct object is a predictive, exploratory model, not
a causal network.

## Answers to the three questions

### 1. Which aspects improve joint trajectory forecasts?

- **Cross-neuron lag structure:** yes, predictively. B3 versus B1 energy is
  `+0.371 [0.128, 0.530]`; 60.6% of landmarks improve.
- **Static covariance / full innovations:** the earlier factorial found a
  negative full-covariance effect (`-0.086` energy). The new B experiments isolate
  the conditional mean with common innovations; the gain is not attributable to a
  different innovation stream.
- **Matched own-neuron ablation:** B4r is an independent AR refit on the same
  500-frame window. B3 beats it by `+0.547 [0.355, 0.698]` energy and by
  `+0.428 [0.191, 0.621]` mean MSE. The B3-versus-B1 MSE contrast is weaker
  because the 200-frame independent AR is a stronger baseline than the
  500-frame one; window length and cross-neuron structure are partly confounded.
- **Temporal alignment null:** B3 beats the circular surrogate by `+1.249
  [0.888, 1.537]` energy. The surrogate preserves marginals and lag-1
  autocorrelation but destroys cross-neuron alignment.
- **Caveat:** B3 versus B1 mean MSE is not conclusive. Energy rewards spread and
  shape as well as the mean. B3 versus the matched 500-frame independent refit
  improves both energy and MSE, but that refit is weaker than B1.

### 2. Are inferred interactions robust?

No, not as an interaction graph.

- Spectral radius is cap-stable (segment CV `0.0015`), but off-diagonal
  coefficient correlation across chronological segments is `0.052`.
- Non-normality is `1.47-2.70`; transient amplification is `3.6-4.7x`. A stable
  radius does not imply a stable transient.
- Complex eigenvalues are common (35-49 of 103), but their periods (4.7-20.3 s)
  do not match the observed slow spectra (dominant periods 252-756 s). No
  oscillation is claimed.
- Synthetic common-input systems produce spurious off-diagonal structure and a
  worse-than-diagonal forecast, so the Kato design cannot separate direct coupling
  from a shared drive.

### 3. How much can be learned about virtual perturbations?

Only a bounded, model-dependent sensitivity statement.

- Six named neurons are present in every session; no imputation.
- A `+1 SD` displacement gives target responses of `0.58-1.26` (9-21% of baseline
  norm), larger than matched controls (`0.05-0.09` median).
- Median target response cosine across ten fits is `0.556`; only AVEL exceeds
  `0.80`. Some fits reverse the sign.
- Median OOD is `1.4%`; maximum is `13.5%`.
- Finite differences agree with the analytic Jacobian to `8.4e-15`.
- No neuron is a validated stimulation target. No biological effect is estimated.

## Gates

| Gate | Status | Evidence |
| --- | --- | --- |
| G1 reproduction | PASS | max difference `8.9e-16`; J2/J5 mean-equivalent |
| G2 cross-neuron information | PASS | B3 beats B1, B4r and circular surrogate on energy |
| G3 operator reliability | FAIL | off-diagonal correlation `0.052` < 0.50 |
| G4 state transition fidelity | FAIL | B3 TV `0.053` worse than Markov `0.031` |
| G5 sensitivity reliability | FAIL | median cosine `0.556`, sign agreement `0.80` |
| G6 empirical support | PASS | median OOD `1.4%` < 5% |
| G7 generative validity | FAIL | frozen calibration and trajectory gates fail |
| G8 biological validation | NOT_EVALUABLE | no matched intervention data |

## Figures

Generated under `reports/simulator/kato_interactions/figures/`:

- `fig_cross_neuron_energy.png` — matched-innovation model comparison;
- `fig_cross_neuron_controls.png` — B3 versus B1, B4r and circular surrogate;
- `fig_surrogate_controls.png` — surrogate energy distributions;
- `fig_operator_spectra.png` — frozen eigenvalue scatter;
- `fig_operator_stability.png` — segment coefficient-of-variation;
- `fig_state_transitions.png` — observed, B3 and B1 transition matrices;
- `fig_sensitivity_agreement.png` — target/control response cosine (warning on figure);
- `fig_sensitivity_ood.png` — OOD support (warning on figure);
- `fig_j2_j5_equivalence.png` — J2/J5 scatter with identity line.

## Negative results that must be preserved

- B3 does not beat the training-only Markov null on transition TV.
- B4 (diagonalized full operator) is a poor mean model and is not a clean
  own-dynamics ablation; B4r is the matched comparator.
- The B5 surrogates are worse than B1; they are negative controls, not realistic
  simulators.
- Session 1 reverses the B3-versus-B1 energy sign.
- J2 and J5 are not per-landmark identical; only their means are equivalent.
- Calibration and trajectory gates remain failed.

## Strongest allowed sentence

In a fixed 10-D PCA fitted on training frames only, the Kato population VAR
provides exploratory evidence of improved joint H20 energy over an independent
per-neuron AR baseline with matched innovations, but the fitted interactions are
not stable enough to identify direct coupling, the state-transition law is not
validated, and virtual perturbations remain model-dependent and causally
unvalidated.
