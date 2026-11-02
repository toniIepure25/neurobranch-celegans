# JOINTDYN trajectory fidelity

**Result: G5 FAIL on neural covariance. The corrected population model beats persistence and independent AR on joint trajectory energy, but its covariance error remains above the frozen 0.5 threshold.**

## 1. Trajectory scores

The trajectory score flattens the 20-frame path in the fixed 10-D PCA. Lower is better.

| Model | H20 trajectory energy | H100 trajectory energy | Neural covariance error | Within-path covariance error | Autocorrelation error |
|---|---:|---:|---:|---:|---:|
| J0 persistence | 2.286 | 3.948 | 0.630 | 0.534 | 0.195 |
| J1 independent diag | 2.435 | 3.857 | 0.505 | 0.511 | 0.174 |
| J2 population diag | **2.058** | 3.820 | 0.572 | 0.524 | 0.160 |
| J3 population full | 2.277 | 4.330 | 0.762 | 0.649 | 0.213 |
| J4 independent full | 2.359 | 3.679 | 0.517 | 0.499 | 0.189 |
| J5 population lead-1 | **2.060** | **3.774** | 0.604 | 0.550 | 0.164 |

J5 improves H20 trajectory energy by 0.226 over persistence and 0.375 over independent AR. The ratio to persistence is **0.901**, which passes the trajectory-energy part of G5.

The covariance part fails: **0.604 > 0.5**. Autocorrelation passes: **0.164 < 0.35**.

## 2. Paired trajectory contrasts

| Comparison | Improvement | 95% interval |
|---|---:|---:|
| J5 vs J1 | 0.375 | [0.218, 0.522] |
| J5 vs J0 | 0.226 | [0.089, 0.337] |
| J2 vs J1 | 0.377 | [0.224, 0.527] |
| J3 vs J1 | 0.159 | [-0.005, 0.309] |

The trajectory advantage over independent AR is supported. The full-covariance model is not.

## 3. Covariance structure

Pooled cross-neuron covariance is computed over all paths and forecast steps in the 10-D PCA and compared by relative Frobenius error.

The observed covariance is better matched by the diagonal population model than by the full-covariance model. The full covariance produces large, poorly identified directions; its error is 0.762, worse than persistence.

This is the central joint-dynamics finding: **adding a full covariance does not recover the observed joint covariance**. The population advantage is in the conditional mean path, not in the innovation covariance.

## 4. Temporal autocorrelation

Mean lag-1 autocorrelation error:

- J5 0.164;
- J1 0.174;
- J0 0.195;
- J3 0.213.

The J5 profile is closer to the observed profile than the baselines. The full lag-1-10 figure is in `figures/jd_autocorrelation.png`.

## 5. State occupancy and recurrence

A six-state k-means model was fitted on training PCA states only. Occupancy is the fraction of H20 steps in each state; total variation is against the observed occupancy.

| Model | Occupancy TV | Recurrence |
|---|---:|---:|
| J0 | 0.153 | 0.958 |
| J1 | 0.145 | 0.963 |
| J2 | 0.096 | 0.949 |
| J5 | **0.096** | 0.946 |
| Observed | — | 0.965 |

The population model reproduces coarse state occupancy better than the independent baseline. Recurrence is slightly too low, consistent with excess switching. These are descriptive statistics, not evidence of biological bifurcations.

## 6. Long-horizon stability

All ensembles are finite at H100.

| Model | Mean maximum absolute PC value | Maximum over landmarks |
|---|---:|---:|
| J0 | 18.8 | 33.7 |
| J1 | 7.1 | 14.9 |
| J2 | 13.8 | 24.2 |
| J3 | 25.1 | 54.4 |
| J5 | 14.5 | 26.2 |

The population models do not explode numerically because the VAR operator is capped at spectral radius 0.995. They can still produce large transient excursions. The full-covariance model is the least stable.

## 7. Visual trajectories are not evidence

The dashboard and the example figure show an actual held-out ensemble and the actual continuation, not a decorative fan. Visual divergence is not interpreted as biological branching.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation: `NOT_EVALUABLE`.

## 8. Gate interpretation

- Trajectory energy: passes relative to persistence.
- Temporal autocorrelation: passes.
- Neural covariance: fails.
- G5 overall: **FAIL**.
- A model cannot be called a validated generative simulator from trajectory energy alone.

## 9. Artifacts

- `scores_by_landmark.csv`
- `score_summary.csv`
- `covariance_h20.csv`
- `autocorr_profile.csv`
- `example_trajectories.csv`
- `example_bands.csv`
- `example_observed.csv`
- `example_metadata.json`
- `figures/jd_example_trajectory.png`
- `figures/jd_covariance.png`
- `figures/jd_autocorrelation.png`
- `figures/jd_trajectories.html`
