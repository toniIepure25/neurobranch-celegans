# Adaptive dynamics and stability: results

## Bottom line

Temporal adaptation helps prospective H20 forecasting relative to the validated
frozen VAR. The prespecified primary model, rolling VAR with a 500-frame window,
reduces paired H20 MSE by 0.657 (33.9%; 95% block-bootstrap interval
[0.326, 0.986]), and 86% of landmarks improve. The advantage appears at every
horizon, but it is not a distributional reconstruction of the neural dynamics
and it is not uniform across sessions: session 2 gets slightly worse, and the
largest covariance/subspace shifts are exactly where adaptation helps least.

Large half-split latent angles are mostly an estimation/occupancy artifact with
a modest real drift superimposed. The calcium rollout has a correctable
feedback defect; correcting it yields finite but underfitted trajectories that
are worse than persistence in four of five sessions.

## Answers to the required questions

### 1. Are the large half-split latent subspace angles evidence of nonstationarity or an estimation artifact?

Mostly an estimation/occupancy artifact, with a modest reproducible drift.
Sliding 256-frame windows have mean angles of 55–64° even in the early part of
the recording. A stationary phase-randomized surrogate with the same covariance
and autocorrelation reaches 52–60° at its 95th percentile and up to ~90° at its
maximum. Every late-window mean exceeds its surrogate 95th percentile, but by
only 2.3–12.0°. A synthetic stationary system gives 8°, while a synthetic
regime switch gives 63°, so the analysis can detect a real switch. The real
recordings show gradual drift, not a clean switch, and the 56–62° half-split
number should not be read as the size of a biological state change.

### 2. Does the calcium observation model have a correctable numerical defect?

Yes. The fitted operators have spectral radii 0.761–0.984, yet the original
rollout reaches 10⁸–10²⁰ by 100 steps. The rollout re-encodes each predicted
calcium value as a new high-pass drive and feeds it back into the latent
recursion. The high-pass gain `(1-alpha)/alpha` plus the PCA inverse map creates
a positive feedback loop. This is an implementation defect, not evidence of an
unstable fitted `A`.

### 3. Can corrected calcium dynamics generate stable autonomous trajectories?

Yes. `causal_calcium_lds_corrected_v1` keeps the latent drive and calcium
recursions separate and never re-encodes predicted calcium. All 175 real test
trajectories and the synthetic control are finite. Corrected H20 MSE is
1.24–3.13 on real sessions, versus persistence 0.58–2.53; it beats persistence
only in session 2, and that interval includes zero. On a known synthetic stable
calcium system, corrected H20/H100 MSE is 0.024/0.024 and beats persistence
(0.045/0.048). Stability is achieved, but predictive credibility is not.

### 4. Does rolling-window VAR outperform frozen VAR?

Yes at the prespecified primary setting: rolling w500 improves H20 MSE by 0.657
[0.326, 0.986], wins 86.3% of landmarks and 4/5 sessions. Rolling w1000 is
better in aggregate (0.752 [0.504, 1.036], 5/5 sessions), but it was not the
prespecified primary model. Rolling w200 is not reliably better (0.207
[−0.377, 0.706], 3/5 sessions).

### 5. Does exponentially weighted VAR outperform frozen VAR?

Yes in aggregate. EWMA h1000 improves H20 MSE by 0.758 [0.523, 1.029] (44.0%),
EWMA h300 by 0.714 [0.394, 1.035], and EWMA h100 by 0.506 [−0.004, 0.911].
EWMA h1000 is the best aggregate model but is post hoc relative to the
prespecified rolling-w500 primary. EWMA h100 has the widest interval and is
borderline.

### 6. Does adaptation improve H20 free-running simulation?

Yes relative to frozen VAR, and modestly relative to persistence. Primary
rolling w500 H20 mean MSE is 0.941 versus frozen 1.597 and persistence 1.077;
mean skill versus persistence is +0.134. EWMA h300 reaches +0.201. Roughly
53.5% of neuron-landmark series beat persistence under rolling w500, compared
with 39.0% under frozen. The improvement is real for the paired frozen
comparison, but half the neurons still do not beat a trivial baseline.

### 7. Which neural components benefit from adaptation?

The benefit is broad but skewed. Median per-neuron H20 skill is +0.158 for
rolling w500 (53.5% >0), +0.194 for EWMA h300 (54.1% >0), and −0.997 for frozen
(39.0% >0). Numeric traces benefit more than named neurons: rolling-w500 median
skill is +0.262 for numeric traces versus −0.239 for named neurons; only 45.9%
of named-neuron rows beat persistence. This is a descriptive subset result, not
evidence that named neurons are causally uninvolved.

### 8. Are improvements concentrated near apparent regime changes?

No. The largest covariance-shift quartile has the **lowest** mean relative
improvement for rolling w500 (−0.032 versus 0.376–0.536 in the other
quartiles). The largest subspace-angle quartile improves 0.135 versus 0.463 in
the smallest. Occupancy shift has a positive linear coefficient (0.533
[0.108, 0.839]) but the quartile pattern is non-monotonic (0.137, 0.449, 0.462,
0.310). Adaptation is therefore not specifically rescuing windows of strong
distributional change; it is broadly tracking local level and damping variance.
Recent frozen error has a negative coefficient (−0.286 [−0.640, 0.089],
including zero).

### 9. Does adaptation improve the match to measured neural temporal statistics?

Only partially. At H20, rolling w500 reduces autocorrelation error from 0.748 to
0.556 and covariance relative error from 1.080 to 1.018, but the predicted
standard-deviation ratio is 0.238 versus 0.360 for frozen, and the OOD fraction
rises from 2.1% to 5.4%. Forecasts are severely under-dispersed. Lower MSE is
achieved largely by shrinking toward a local mean. Gate D fails.

### 10. Are improvements stable across sessions and evaluation horizons?

Across horizons, yes: adaptive models are better at H1 through H100. Across
sessions, no: the primary rolling w500 improves four of five sessions, but
session 2 has a negative paired improvement of −0.327 [−1.212, 0.414]. The
aggregate is strongly influenced by sessions 0 and 1. EWMA h1000 and rolling
w1000 win all five sessions in aggregate, but session 2 remains weak
(0.165 [−0.148, 0.500] for EWMA h1000). No animal-level replication is possible
because only session identities are known.

### 11. What tradeoffs exist between adaptive forecasting and long-run stability?

Adaptive spectral radii are capped at 0.995; realized means are 0.990–0.994.
Nineteen to 47 percent of refits required clipping. No trajectory became
non-finite. At H100, adaptive MSE is 1.34–1.46 versus frozen 1.98, but all
models decay toward low-variance trajectories. Faster adaptation (w200) has the
worst long-horizon error (2.35 at H100). Stability is maintained by strong
shrinkage, at the cost of variance and distributional fidelity.

### 12. Are any virtual perturbation predictions more consistent across models?

No. Across 75 descriptive frozen-versus-adaptive impulse trajectories, the
median cosine is 0.060 (range −0.162 to 0.570). Agreement is weak and
neuron-dependent. No neuron is ranked. Every result is:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

### 13. What remains unresolved because of missing experimental observations?

- No independent animals: five repeated sessions cannot establish biological
  replication.
- No randomized oxygen schedule or sham in WT_NoStim, so oxygen transitions
  cannot be separated from elapsed time here.
- No direct voltage/calcium state measurement, so latent state and observation
  model remain confounded.
- No experimental perturbation matched to model impulses, so perturbation
  directions are not causally identifiable.
- No intervention data, so `BIOLOGICAL_INTERVENTION_VALIDATION` is
  `NOT_EVALUABLE`.

### 14. What scientific result would be strongest to present to Denis Kovačević?

The paired frozen-versus-adaptive result with its uncertainty and its negative
distributional finding: adaptation cuts H20 error by about one third, but it
does so by under-dispersing forecasts, fails on the session with the largest
covariance shift, and does not make virtual perturbations agree. The calcium
feedback defect is a clean, reproducible implementation result; the
nonstationarity audit is a caution against over-reading PCA angles.

### 15. What small independent experiment could validate the findings?

A pre-registered within-session comparison in independent animals with
randomized oxygen timing plus a sham schedule, using the same prediction
landmarks and a held-out animal split. The decisive test is whether local
adaptation still improves H20 error and variance fidelity when the timing of the
environmental change is randomized and the evaluation animal has not been used
for any model selection.

## Decision gates

| Gate | Status | Evidence |
|---|---|---|
| A Nonstationarity characterized | **PASS** | Matched surrogate and synthetic controls separate a modest drift from a large estimation/occupancy baseline. |
| B Numerical stability established | **PASS** | Corrected calcium recursion finite on all real and synthetic trajectories; original feedback defect localized. |
| C Adaptive model advantage | **PASS** | Primary rolling w500 paired H20 improvement 0.657 [0.326, 0.986], 4/5 sessions. |
| D Distributional improvement | **FAIL** | Std ratio 0.238, covariance error 1.018, OOD 5.4%; improvement is shrink-to-mean. |
| E Perturbation interpretation | **MODEL-DEPENDENT ONLY** | Median cross-model cosine 0.060; no causal validation. |
| F Independent biological validation | **NOT_EVALUABLE** | No independent animals or matched intervention. |

## Negative findings

- Adaptation does not rescue the largest distributional shifts; those windows
  improve least.
- Session 2 is not improved by the primary model.
- Roughly 46% of neuron-landmark series still fail to beat persistence.
- Forecasts are much too smooth; covariance error remains about the size of the
  signal covariance.
- Corrected calcium dynamics are stable but worse than persistence in four of
  five real sessions.
- Frozen and adaptive perturbation directions disagree almost completely.
- The large PCA angles are not a clean nonstationarity measure.

## Reproducibility and provenance

- Branch: `research/adaptive-dynamics-stability`
- Starting commit: `c8f4f53` (`research/generative-failure-diagnostics`)
- Data: Kato WT_NoStim, five sessions; SHA256
  `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a`.
- No raw data or model weights committed.
- Artifacts: `reports/simulator/adaptive_dynamics/`.
- Manifest: `reports/simulator/adaptive_dynamics/manifest.json`.
- Evidence tier: exploratory, repeatedly inspected Kato data; not external
  confirmation.

## Reproduction commands

```powershell
cd D:\ComputaCenter\NeuroBranch\neurobranch-celegans
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'
python scripts/adaptive_nonstationarity.py
python scripts/adaptive_calcium.py
python scripts/adaptive_models.py
python scripts/adaptive_indicators.py
python scripts/adaptive_perturbation.py
python scripts/adaptive_figures.py
python scripts/adaptive_manifest.py
python -m pytest -q
python -m ruff check .
python -m mkdocs build --strict
```
