# Phase 6 VAR and switching dynamics

## Regularized VAR

For lag \(p\):

\[
\hat{x}_{t+1}=A\,[x_t;\ldots;x_{t-p+1}]+b.
\]

Fitting uses training starts only, standardizes each input column, applies ridge to the design matrix (not the intercept), and optionally scales \(A\) so the companion spectral radius is at most 0.995.

A cap is a numerical stability device, not a biological constraint. It can only damp a fitted model; it does not make an unstable model identifiable.

### Stochastic VAR

The deterministic mean is augmented with innovations:

\[
x_{t+1}=\hat{x}_{t+1}+\varepsilon_t,\qquad
\varepsilon_t\sim\mathcal N(0,\Sigma)
\]

or a residual bootstrap draw. \(\Sigma\) is the Ledoit–Wolf shrinkage covariance of training one-step residuals. The i.i.d. and moving-block bootstraps preserve the empirical residual marginal and, for blocks, some serial dependence.

These innovations are **not** estimated process noise, and adding them necessarily worsens realized-trajectory MSE at H=20. They are judged by distributional and ensemble metrics, not by sample MSE alone.

Implementation: `src/neurobranch/dynamics/models.py::VARModel`.

## Switching VAR

\[
r_t\sim \mathrm{Categorical}(P[r_{t-1}]),\qquad
x_{t+1}\sim\mathcal N(A_{r_t}x_t+b_{r_t},\operatorname{diag}(v_{r_t})).
\]

Fitting uses EM on contiguous training runs, with KMeans initialization on one-step changes and five restarts. Transitions are counted within runs; a transition is never invented across a purged boundary.

### Deliberate restriction

The fitted dynamics matrix is **diagonal** per regime. Full \(100\times100\) switching VARs produced large transient amplification even when the spectral radius was capped, with validation MSE 4–60× persistence. Diagonal dynamics are a compact, stable restriction and are recorded in `diagnostics.diagonal_dynamics=true`.

This restriction means the switching model does **not** test cross-neuron switching coupling. It is a regime-dependent per-neuron autoregression, not a mechanistic network model.

### Result

All switching configurations failed validation skill (K=2 median −3.40; K=3 median −3.31). K=3 was selected only because its median was less negative and occupancy exceeded 5%; it is still a failed model.

Implementation: `src/neurobranch/dynamics/switching.py`.

## What is supported

- A stable one-step VAR with real observed histories can be rolled out for 100 steps without divergence.
- A compact switching VAR can be fitted and sampled.
- Neither construction establishes a discrete worm regime, a bifurcation, or a network mechanism.
