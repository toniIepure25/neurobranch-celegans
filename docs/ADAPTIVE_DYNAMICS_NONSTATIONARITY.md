# Nonstationarity audit

## What was tested

Sliding 256-frame PCA windows (step 64, 10 components) were compared with a
train-only reference subspace, covariance, amplitude, occupancy, and local AR(1)
operator. The null is multivariate phase randomization, which preserves each
neuron's power spectrum and the cross-spectrum while removing temporal
nonstationarity. A stationary synthetic VAR system and a synthetic system with a
mid-recording change in both latent operator and observation loadings were
passed through the identical pipeline.

All angles below are mean principal angles in degrees over 10 components. A
large angle is not by itself evidence of a biological state transition.

## Real recordings

| Session | Early angle | Late angle | Max window mean | Late covariance change | Late amplitude shift | Late occupancy shift | Late local AR drift | Surrogate 95th percentile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 55.19 | 62.67 | 70.23 | 0.978 | 0.649 | 0.720 | 0.962 | 59.76 |
| 1 | 55.69 | 62.59 | 73.64 | 1.188 | 0.547 | 0.706 | 1.007 | 55.34 |
| 2 | 57.53 | 64.21 | 74.55 | 1.927 | 0.471 | 0.670 | 1.070 | 52.22 |
| 3 | 61.59 | 63.74 | 68.63 | 0.942 | 0.710 | 0.659 | 0.975 | 60.35 |
| 4 | 57.34 | 62.89 | 71.41 | 1.072 | 0.479 | 0.549 | 1.067 | 54.95 |

Every session's late-window mean exceeds the 95th percentile of its stationary
surrogate window distribution, but the margin is modest (about 2.3–12.0°), and
the early-window angles are already 55–62°. The observed maxima are 69–75°,
while stationary surrogate maxima reach about 90°. The surrogate comparison
therefore does **not** support a clean transition from one stable subspace to
another. It supports a gradual, moderate drift superimposed on a very large
finite-sample/occupancy baseline.

Covariance changes are of the same order as the reference covariance (relative
Frobenius 0.94–1.93), amplitude shifts are large (0.47–0.71 in mean absolute
log ratio), and local AR operators change by roughly their own norm. These are
not small perturbations, but they are also what would be expected from slow
fluorescence drift, changing state occupancy, and windows with unequal signal
variance.

## Synthetic controls

| Condition | Early angle | Late angle | Late covariance change | Max window mean |
|---|---:|---:|---:|---:|
| Stationary VAR + fixed loadings | 7.54 | 8.23 | 0.463 | 9.14 |
| Regime switch in dynamics and loadings | 16.19 | 62.82 | 1.218 | 63.69 |

The stationary control stays near its reference orientation. The regime control
is detected clearly. The real recordings resemble the regime control in raw
angle magnitude, but their within-session trajectory is much less abrupt and
their surrogate null is also high. The correct conclusion is **mixed**:
nonstationarity is present, but the previously reported 56–62° half-split angles
are not a clean estimate of its size.

## Interpretation

1. The large half-split angles are **not** strong evidence of discrete
   nonstationarity. Window length, finite sample size, state occupancy, and
   unequal variance inflate principal angles even under a stationary surrogate.
2. There is reproducible evidence of gradual drift: late windows are
   consistently farther from the train reference than matched stationary
   surrogates, and synthetic stationary data do not show that pattern.
3. The drift is not a single abrupt regime change. A model with one discrete
   switch is not supported by these trajectories.
4. No anatomical or biological interpretation is attached to PCA axes. A PCA
   rotation is a coordinate choice, not a discovered neural state.

## Consequence for modelling

The audit supports **time-varying local linear estimation** as a candidate, but
not a claim that nonstationarity is the sole cause of forecast failure. It also
argues against fitting a single global latent encoder and calling its axes
stable. Local latent models were therefore not added before the frozen/adaptive
VAR comparison was resolved.
