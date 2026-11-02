# Adaptive dynamics: limitations

## Data limitations

1. **No independent animals.** The five WT_NoStim recordings are repeated
   sessions. All uncertainty is over landmarks and sessions, not animals. A
   session-level win is not biological replication.
2. **Repeated inspection.** These recordings and the chronological test tails
   have been analysed in earlier phases. The test set is not untouched, and the
   experiment is exploratory.
3. **Chronological test tail.** The split is temporal, which is appropriate for
   forecasting, but it also confounds time with any slow drift, animal state, or
   acquisition change.
4. **No randomized environmental intervention.** WT_NoStim has no measured
   oxygen schedule. WT_Stim has a fixed schedule repeated across sessions and
   was not used here; oxygen timing remains confounded with elapsed time.
5. **No direct state measurement.** Fluorescence is a calcium proxy. The latent
   drive and the calcium observation model are not separately identified
   without voltage, spikes, or calibrated calcium.

## Method limitations

- The adaptive models are linear lag-one VARs. They can track local mean and
  covariance but cannot represent nonlinear dynamics, discrete states, or
  long-range dependencies.
- The spectral cap at 0.995 enforces stability by scaling the operator
  uniformly. It is not a probabilistic constraint and it changes the fitted
  dynamics. Nineteen to 47 percent of adaptive refits required clipping.
- Ridge penalties and window lengths were fixed before test evaluation. No
  test-set selection was performed, but the primary model is only one of six
  candidates; the best aggregate model (EWMA h1000) is post hoc.
- Forecast MSE is dominated by high-variance neurons. Per-neuron skill has
  near-zero denominators for some series, producing extreme values; medians and
  positive fractions are reported instead of means.
- Covariance error is computed at the final horizon and can be unstable when
  the number of forecast steps is small. H1 covariance is undefined and is
  excluded.
- The nonstationarity null preserves stationary covariance and autocorrelation
  but not nonlinear structure. It cannot prove that a window is stationary; it
  only calibrates how large an angle can arise from finite sampling and
  stationary temporal structure.
- Principal angles are invariant to rotation and permutation but depend on
  window length, dimensionality, and the reference period. Different choices
  would change the absolute values.
- The synthetic regime switch was designed after observing that a
  latent-only switch was not detectable in observation space; that correction
  is documented, but it means the control is illustrative rather than an
  independent preregistered benchmark.

## Interpretation limits

- A lower MSE does not establish a better generative model. The adaptive
  forecasts are under-dispersed and fail the distributional gate.
- A model-dependent impulse is not a biological intervention. Cross-model
  cosine 0.060 means the perturbation predictions are not robust.
- Named-neuron subset differences are descriptive. They do not identify causal
  neurons or circuits.
- The corrected calcium model's stability is an implementation property, not
  evidence that the latent dynamics are biologically correct.
- The five-session aggregate cannot be generalized to the species, to other
  worms, or to other recording conditions.

## What would change the conclusions

- An animal-disjoint replication with the same prespecified comparison.
- Randomized oxygen timing with a sham schedule and independent animals.
- Direct voltage or spike measurements that separate latent drive from calcium
  observation.
- A matched experimental perturbation that can test model impulse directions.
- A nonlinear or switching model that improves both MSE and distributional
  fidelity, not MSE alone.
