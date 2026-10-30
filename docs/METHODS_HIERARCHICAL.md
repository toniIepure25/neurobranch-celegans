# Hierarchical measurement model: methods

This note documents the Phase 12 model. It is a measurement model for atlas
reliability, not a causal model, a digital twin, or a foundation model.

## 1. Input construction

`cell_summaries` produces one row per animal, target, and neuron with an event
count `n`, mean response `mean`, and within-cell variance `var`. The variance of
a cell mean is `var / n`.

For each target:

1. keep cells with at least one event (`min_events=1` in the primary fit;
   `min_events=2` is a preregistered sensitivity);
2. pivot to an animal x responder matrix of means and sampling variances;
3. require at least three responders and at least `max(min_animals, 3)` animals
   (primary `min_animals=6`);
4. keep responders observed in at least the requested fraction of animals
   (primary: 0.8);
5. drop animals with no remaining observation;
6. subtract each animal's mean across its observed responders
   (`center_animals=True`);
7. standardize the remaining finite values by their pooled standard deviation
   and rescale the sampling variances by the squared scale.

Step 6 is important. Without animal centering, stable animal-wide offsets are
absorbed into the between-target term and inflate `R(22)`. The uncentered fit
gave 0.275 pooled reliability and 0.525 ceiling; the centered primary fit gave
0.304 and 0.551. Neither is adopted because the gates fail.

Cells with a genuine zero variance receive the explicit floor `1e-8` after
standardization and are counted in `n_variance_floored`. Missing or negative
sampling variances are not silently replaced: the target is excluded with
`invalid_sampling_variance`. Missing cells are never imputed, and the mask is
preserved in the likelihood. The floor matters descriptively: with one event per
cell most within-cell variances are exactly zero, so a large fraction of cells
use the floor. Phase 12R reports that fraction and the `min_events=2` sensitivity
rather than treating the known-variance assumption as automatically satisfied.

## 2. Likelihood

For target $r$:

$$
y_{ar} = \mu_r + u_{ar} + e_{ar}, \qquad
\mu_r \sim N(0,\sigma^2), \quad
u_{ar} \sim N(0,\tau^2), \quad
e_{ar} \sim N(0,v_{ar}).
$$

Integrating out $\mu_r$ gives, for each responder, a rank-one covariance across
animals:

$$
\operatorname{Cov}(y_{\cdot r}) = \tau^2 I + \operatorname{diag}(v_{\cdot r})
+ \sigma^2 \mathbf{1}\mathbf{1}^\top.
$$

The code evaluates the Gaussian log-likelihood with the matrix-determinant lemma
and a Woodbury inverse, so the cost is linear in animals and responders. The
prior is

$$
\sigma \sim \operatorname{HalfNormal}(1), \qquad
\tau \sim \operatorname{HalfNormal}(1)
$$

on the standardized scale. The prior is a regularization choice, not a
substantive biological prior.

## 3. Posterior computation

The posterior is evaluated on a deterministic 80 x 80 grid over
$(\log\sigma,\log\tau)$ from $-6$ to $\log 20$. This replaced a random-walk MCMC
pilot whose R-hat was about 1.30 and whose minimum effective sample size was 18
on weakly identified targets. The grid is reproducible and exposes boundary
mass; its cost is discretization.

For each target, the fit reports:

- posterior draws of $(\sigma^2,\tau^2)$, obtained by sampling grid cells with
  probability proportional to posterior mass and adding within-cell jitter;
- `edge_mass`: posterior mass on the grid boundary (gate: < 0.05);
- `n_eff`: effective number of grid cells;
- `R(22)`, its 95% quantile interval, and the ceiling $\sqrt{R(22)}$;
- `R(1)` for the single-animal profile.

The pooled estimate is the median across eligible targets, not a mixture over
targets. No target-level ranking is published.

## 4. Reliability estimand

$$
R_r(n) = \frac{\sigma_r^2}{\sigma_r^2 + (\tau_r^2 + \nu_r)/n},
\qquad \nu_r = \frac{1}{|\text{observed cells}|}\sum_{a,r} v_{ar}.
$$

$R_r(n)$ is the reliability of a profile averaged over $n$ animals under the
model. It is **not** the correlation between two observed halves, and it is
conditional on the observed coverage pattern. The ceiling is
$\sqrt{R_r(22)}$, the attenuation limit for a correlation with a noisy profile.

The 22-animal value is inherited from the Phase 6--8 fold size. It is not a
recommended sample size; `design_posterior.csv` is a model-conditional planning
table and is not prospectively validated.

## 5. Calibration and adversarial checks

- **SBC** (`simulation_based_calibration`): 1,000 replicates, three targets each,
  real coverage masks and sampling variances, $\sigma,\tau$ drawn from the
  fitting prior, 2,000 posterior draws on an 80 x 80 grid. Simulated data and the
  fit use the same animal centering; the truth uses the centered effective
  variance. Ranks and 95% coverage are computed at $n=1$ and $n=22$; a KS test
  checks rank uniformity and a Wilson interval is reported.
- **Held-out PPC** (`heldout_predictive_check`): twenty animal-level splits, up to
  40 targets, 1,200 draws on a 60 x 60 grid; intervals from the training fit are
  compared with held-out animal profile correlations after the same centering.
- **Negative control** (`negative_control`): each animal's responder labels are
  permuted exactly within that animal, destroying true cell correspondence while
  preserving the coverage pattern, marginal distributions and known variances.
  One hundred targets are requested; requested, eligible, excluded and reasons
  are reported. A calibrated model should return near-zero reliability.
- **Prior sensitivity** (`prior_sensitivity`): the same fit at HalfNormal scales
  0.5, 1.0, and 2.0 on a 50 x 50 grid. A range above 0.05 means the conclusion is
  prior-driven.

The Phase 12 run used a smaller SBC (200 replicates) and an unaligned truth
formula; its calibration numbers are historical and are superseded by Phase 12R.
The point estimate is unaffected by the alignment correction.

## 6. Known limitations

1. The additive Gaussian model may not describe sparse, heavy-tailed calcium
   responses; the input response distribution has a standard deviation of 23.8
   and large outliers.
2. Most cells contain one event, so their within-cell sampling variance is
   exactly zero. The fit uses an explicit `1e-8` floor and reports the floored
   fraction; this is an approximation, not a measured variance. The
   `min_events=2` sensitivity is reported because it changes both eligibility and
   the pooled estimate.
3. The Phase 12R negative control showed positive null bias: median $R(22)=0.120$
   and q95 = 0.594 on permuted profiles (48 eligible of 100 requested). This is
   the primary adversarial check; a positive null remains a model limitation.
4. Phase 12R SBC rank uniformity failed (KS p = 1.43e-06) and held-out predictive
   median absolute error was 0.207; coverage alone (0.933 SBC, 0.904 PPC) does
   not make the interval calibrated.
5. The deterministic grid is not a full posterior sampler; grid resolution and
   boundary mass are reported but do not fix model misspecification.
6. The 0.8 coverage filter changes the responder population. The estimate applies
   to broadly observed responders, not to all responders.
7. The model does not identify causal effects, neural mechanisms, or
   intervention responses. Perturbation predictions remain model-dependent.

## 7. External methods anchors

The variance-decomposition and calibration design follow the methods records in
`research/evidence.csv`: Shrout & Fleiss (1979; DOI
`10.1037/0033-2909.86.2.420`) for reliability as a variance ratio, and Modrak et
al. (2025; DOI `10.1214/23-ba1404`, PMID 41049725) for simulation-based
calibration. These are methods anchors, not evidence that this model is
calibrated. No new literature search was performed for Phase 12; the negative
result is internal to this atlas and this model.

## 8. Code map

| Purpose | Path |
|---|---|
| Fit and posterior summaries | `src/neurobranch/benchmark/hierarchical.py` |
| SBC, PPC, negative control, prior sensitivity | `src/neurobranch/benchmark/hierarchical_calibration.py` |
| Cluster bootstrap and Holm/MDE | `src/neurobranch/benchmark/inference.py` |
| Posterior design table | `src/neurobranch/design/posterior_design.py` |
| Artifact generation | `scripts/phase12_hierarchy.py`, `scripts/phase12_benchmark.py`, `scripts/phase12_manifest.py` |
| Contract tests | `tests/benchmark/test_hierarchical.py`, `tests/benchmark/test_hierarchical_calibration.py`, `tests/benchmark/test_inference.py`, `tests/test_phase12_reports.py` |
