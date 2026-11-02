# Probabilistic adaptive dynamics: results

## Bottom line

The deterministic adaptive VAR improvement does **not** support a calibrated
probabilistic simulator in this parameterization. The conditional mean is
accurate, but the stochastic extension built from its own prequential
innovations is grossly overdispersed. The primary model is worse than
persistence, a rolling local mean and an independent per-neuron AR on proper
scores and calibration.

The stronger scientific finding is earlier: the full adaptive population VAR has
no demonstrated advantage over simple adaptive baselines. Its deterministic gain
over the frozen VAR is almost entirely local level tracking.

**Evidence tier: exploratory repeated-session reanalysis.** The five recordings
are repeated sessions, not confirmed independent animals. Landmarks overlap and
are not biological replicates. No interventional data were used.

## Stage results

| stage | result | gate |
|---|---|---|
| A reproduction | exact to `1.1e-13`; 41.1% aggregate and 33.9% per-landmark relative improvement reconciled | PASS |
| B baseline ablation | B5 vs local mean `+0.026` `[−0.129, +0.152]`; vs independent AR `+0.011` `[−0.080, +0.078]` | FAIL |
| C residual diagnostics | strong lag-1 autocorrelation `0.57`, cross-neuron covariance, nonstationary scale; block bootstrap selected on train only | DESCRIPTIVE PASS |
| D stochastic scores | primary energy H20 `4.605` vs local mean `3.263` and independent AR `3.077` | FAIL |
| E calibration | coverage 50/80/95 = `0.904/0.992/0.999`; spread/error `14.85` | FAIL |
| F trajectory fidelity | total std ratio `3.40`; OOD `0.380`; 95% width `9.97` | FAIL |
| G trajectory example | descriptive fan; not a branching claim | model-dependent only |
| H perturbation | not launched because D–F failed | NOT_EVALUABLE |

## Answers to the 18 report questions

### 1. Did the prior deterministic result reproduce exactly?

Yes. At the same 175 landmarks, the maximum absolute difference from the
historical frozen and rolling H20 MSEs was `1.1e-13` and `4.7e-13`,
respectively. The frozen mean was `1.597159`, rolling w500 was `0.940602`, and
the mean paired difference was `0.656557`.

### 2. Why were 41.1% and 33.9% both reported?

They are different estimands. `41.1%` is `1 − mean(adaptive)/mean(frozen)`, the
ratio of aggregate MSEs. `33.9%` is the equal-weight mean of per-landmark
relative improvements, `mean[(frozen−adaptive)/frozen]`. Both are arithmetically
correct; neither is a biological effect size.

### 3. Does the adaptive population VAR beat a rolling local mean?

No. The paired H20 improvement is `+0.0265` with a 95% block-bootstrap interval
`[−0.1292, +0.1518]`. The interval includes zero and the effect is small relative
to the model error.

### 4. Does it beat independent per-neuron AR?

No. The paired improvement is `+0.0110` `[−0.0798, +0.0775]`. Only 57.7% of
landmarks favor the full VAR, essentially a coin flip.

### 5. Does it beat persistence?

Yes, modestly: `+0.1195` `[+0.0167, +0.2058]`. But simpler adaptive baselines
also beat persistence, so this does not isolate cross-neuron dynamics.

### 6. What do the innovations look like?

They are non-Gaussian, heavy-tailed, cross-neuron correlated and temporally
dependent. The adaptive residual trace is `82.5`; mean absolute cross-neuron
correlation is `0.108`; effective rank is `27.4` of roughly 103 predictors;
off-diagonal covariance fraction is `0.743`.

### 7. Are the residuals white?

No. Training median absolute lag-1 autocorrelation is `0.584`, with lag-5
`0.514` and lag-10 `0.397`. Independent AR residuals are much closer to white
(`0.108` at lag 1), which is consistent with the independent model absorbing
per-neuron dynamics.

### 8. Is residual variance stationary?

No. Relative covariance drift across thirds is `1.49`, and residual norm
correlates `0.379` with recent state norm. A single frozen covariance is an
approximation.

### 9. Which innovation model was selected, and how?

The training-only rule selected the moving-block bootstrap because the median
absolute lag-1 adaptive residual autocorrelation (`0.570`) exceeded `0.1`. The
primary model is P6 with block length 10. No test residual was used.

### 10. Did the stochastic adaptive model improve the proper score?

No. Primary energy at H20 is `4.605`, worse than persistence (`2.948`), local
mean (`3.263`) and independent AR (`3.077`). Paired improvements are all
negative with intervals excluding zero.

### 11. Is it calibrated?

No. 50/80/95% coverage is `0.904/0.992/0.999`; PIT calibration error is `0.085`;
mean spread/error is `14.85`. The intervals are too wide and cover almost
everything.

### 12. Does it preserve the observed variance?

No. Within-ensemble variance is `11.46` versus observed `0.861`. Total implied
variance is `12.12`, a standard-deviation ratio of `3.40`.

### 13. Does it preserve covariance and temporal structure?

Not credibly. The 10-D temporal covariance error is near `1.0`, the variogram
score is worse than every baseline (`0.694`), and the autocorrelation error is
`0.521`. The 10-D truth covariance is nearly singular, so the covariance error
is not a precise discriminator; the variance ratio and coverage are decisive.

### 14. Is it stable at long horizons?

It remains numerically finite, but fidelity degrades. Mean energy rises from
`2.007` at H5 to `4.605` at H20, `7.387` at H50 and `8.838` at H100. Stability
is not accuracy.

### 15. Does it outperform probabilistic baselines?

No. It is worse than P0, P1, P2 and P3 on energy, CRPS and variogram. A post-hoc
diagonal variant (P7) reaches energy `2.838`, better than the local mean but not
reliably better than independent AR (`+0.239` `[−0.018, +0.470]`), and it is
still overdispersed (95% coverage `0.986`, total std ratio `1.87`). P7 is not a
gate pass.

### 16. Can it support trajectory simulation?

Only descriptively, and not as a calibrated generator. The example ensemble mean
is close to the observed continuation, but its population-mean spread is about
`4.7×` its error. The full-space variance is `3.4×` too large. A visually
plausible fan is not evidence of generative fidelity.

### 17. Can it support a biological perturbation claim?

No. No perturbation screen was launched. The model failed the calibration and
variance-fidelity gates, so a virtual perturbation would be
model-dependent and not causally interpretable. Status: `NOT_LAUNCHED`,
`NOT_EVALUABLE`.

### 18. What is the next credible experiment?

Test whether the failure is specific to the innovation parameterization before
building a larger model:

1. fit a state- or horizon-conditioned innovation model that targets the
   observed H20 covariance directly, with strict train/validation separation;
2. compare diagonal, low-rank-plus-diagonal and full shrinkage covariance under
   identical mean operators;
3. evaluate on held-out sessions and a genuinely unseen recording;
4. only then consider perturbation propagation.

See `PROBADAPT_NEXT_EXPERIMENTS.md`.

## What is supported

- The deterministic adaptive VAR is more accurate than the frozen VAR on these
  landmarks.
- Local level adaptation, not demonstrated cross-neuron coupling, explains most
  of that gain.
- The innovations are strongly non-iid and nonstationary.
- A stochastic extension built from those innovations is not calibrated.
- A diagonal-innovation model is a useful lead but is not validated.

## What is not supported

- A calibrated generative simulator of future neural activity.
- A biological branching or bifurcation claim.
- A causal intervention claim.
- Animal-level generalization from five repeated sessions.
- The claim that cross-neuron dynamics are absent in the animal; the claim is
  only that this model and split do not demonstrate added predictive
  information.

## Execution and verification

- Stage A reproduction: `scripts/probadapt_reproduction.py`.
- Stage B ablation: `scripts/probadapt_baselines.py`.
- Stage C diagnostics: `scripts/probadapt_residuals.py`.
- Stage D–F evaluation: `scripts/probadapt_main.py`.
- Stage G example: `scripts/probadapt_examples.py`.
- Figures: `scripts/probadapt_figures.py`.
- Dashboard tab: `app/probabilistic_ui.py`.

Tests added in `tests/test_probabilistic_adaptive.py` cover score propriety,
coverage, PSD covariance, centering, seed reproducibility, block dependence,
Monte Carlo mean recovery, recursive rollout, horizon alignment, long-horizon
finiteness, leakage boundaries and cross-neuron synthetic information.

## Reproduction

```powershell
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'
python scripts/probadapt_reproduction.py
python scripts/probadapt_baselines.py
python scripts/probadapt_residuals.py
python scripts/probadapt_main.py
python scripts/probadapt_examples.py
python scripts/probadapt_figures.py
```

Artifacts are under `reports/simulator/probabilistic_adaptive/`. No raw
recordings or model weights are committed.

## Commit history

The branch contains logical commits for reproduction, ablation, diagnostics,
stochastic evaluation, trajectory example/dashboard, documentation and
verification. Exact hashes are reported in the final response.
