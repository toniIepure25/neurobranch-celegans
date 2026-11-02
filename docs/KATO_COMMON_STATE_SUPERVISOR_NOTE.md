# Supervisor note: Kato common-state controls

**Question.** Does cross-neuron history retain prospective predictive information
after accounting for own-neuron history, local drift, shared population activity
and a low-dimensional common-factor control?

**Data.** Five Kato 2015 `WT_NoStim` recordings only; 15,662 frames; 175 reused
H20 landmarks; native dimensions 103/127/123/118/121. No WT_Stim, external data,
new architecture, large hyperparameter search or perturbation screen was used.

**Design.** Frozen protocol and matched ladder: B0 persistence; B1m matched own AR
(200); B2 plus linear drift; B4m matched own AR (500); B5-k own AR plus
training-only PCA factors (`k=1,3,5,10`, primary 10); B3 full population VAR.
Historical B1h/J2/J5/B4r are reproduction checks only. Primary endpoint is paired
H20 PCA-10 energy, `B5 - B3`, positive when B3 is better; moving-block bootstrap
within session, block 5 (blocks 3 and 10 sensitivity). All models reuse the same
J5 innovations.

**Result.** G1-G6 pass; G7 fails; G8 is `NOT_EVALUABLE`.

- B3 energy `2.700` vs B5-10 `2.965`: advantage `+0.264`
  `[+0.075, +0.403]`, B3 better at 64.0% of landmarks.
- Population-over-B4m gain is `+0.540`; common factors account for `+0.275`
  (51.0%) and residual cross-neuron information for `+0.264` (49.0%).
- The advantage is horizon-dependent: B5 is better at H1; B3 is better at
  H5-H20. B3 also has better 95% coverage (0.938 vs 0.725).
- Session 2 reverses; sessions 0, 1, 3 and 4 are positive.
- Neuron-space energy agrees in direction (`+0.162`) but its interval includes
  zero. MSE is mixed.
- B3 beats the common-preserved residual-shift surrogate by `+0.557`
  `[+0.316, +0.768]`; the result is not reproduced by arbitrary residual
  arrangement.
- Synthetic direct-coupling systems give positive B3-B5 separation in all six
  seed-scenario combinations; common-only and time-varying-common are negative.
  G5 passes.

**Interpretation.** There is residual conditional predictive information beyond a
ten-factor statistical common-state control. It is modest, horizon-dependent and
not uniform across repeated sessions. It is compatible with an unmeasured common
input, nonlinearity, a state variable, or estimator bias. It is **not** evidence
of direct coupling, synapses, a command neuron, or a biological intervention
target.

**Decision.** Preserve as an exploratory positive predictive result with a firm
structural caveat. Do not start another modelling phase. The next defensible
step would require genuinely held-out animals or sham-controlled intervention
data, neither of which is available here.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
Biological intervention validity: `NOT_EVALUABLE`.

**Artifacts.** `reports/simulator/kato_common_state/`, `scripts/kato_common_state_*`,
`docs/KATO_COMMON_STATE_*.md`, `tests/test_kato_common_state.py`.
