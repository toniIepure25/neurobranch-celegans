# Kato common-state controls: models and evaluation

This note documents the fixed model ladder used in
[the protocol](KATO_COMMON_STATE_PROTOCOL.md). It is written after the protocol
was frozen and before the final Kato comparison was interpreted.

## Why a new control ladder was needed

The previous interaction audit compared a full population VAR (B3) with an
independent AR (B1) and reported a B3 advantage. That contrast mixed three
changes at once: population cross-terms, a longer fitting window, and a much
weaker penalty on the independent model. The historical B1 adds `1` to a sum of
squared centred values over roughly 200 transitions, while B3 applies ridge
`1.0` to standardized features. A reanalysis therefore needs a matched
own-neuron baseline before the B3 advantage can be interpreted.

The matching is implemented in
`src/neurobranch/kato_common_state/models.py`. The historical models are left
untouched so that the original numbers remain reproducible.

## Fixed ladder

| ID | Model | Window | Penalty | Purpose |
| --- | --- | ---: | --- | --- |
| B0 | persistence | none | none | lower bound |
| B1h | historical independent AR | 200 | weak effective ridge | exact reproduction only |
| B1m | matched independent AR | 200 | ridge `1.0` on standardized design | fair own-history baseline |
| B2 | B1m + local linear time trend | 200 | ridge `1.0` | global drift control |
| B4m | matched independent AR | 500 | ridge `1.0` | matched-window own-history baseline |
| B5-k | B4m + `k` factor scores | 500 | ridge `1.0` | common-factor control |
| B3 | full population VAR | 500 | ridge `1.0` | population model |
| B6 | B3 with zeroed off-diagonals | 500 | inherited | coefficient ablation diagnostic |

B1h, J2 and J5 are historical checks only. The inferential ladder uses B0, B1m,
B2, B4m, B5 and B3. B6 is never presented as a refitted restricted model.

## Common-factor construction

For each session, PCA is fitted on training frames only:

```text
z_t = x_t - mean_train
f_t = P_k^T z_t
common_t = P_k f_t
residual_t = z_t - common_t
```

`k` is fixed to `{1, 3, 5, 10}`. The primary control is `k=10`, the largest
prespecified factor count. No factor count is selected on the test block.

B5 fits a factor VAR on the same causal 500-frame window and then, for each
neuron, regresses its next value on its own current value and the current factor
scores. This is a diagonal-plus-rank-`k` approximation, not an anatomical latent
variable model. The factors are statistical controls for shared activity, not
identified common inputs.

## Evaluation

The primary endpoint is the paired H20 multivariate energy advantage of B3 over
B5-10. Energy is lower-is-better, so the positive advantage is `B5 energy - B3
energy`; positive means the population model is better. The interval is a
moving-block bootstrap within session (primary block 5; blocks 3 and 10 as
sensitivity). The five sessions are repeated recordings, so the interval is not
an animal-level interval.

Secondary endpoints are conditional-mean MSE, joint trajectory energy, coverage,
neural covariance and autocorrelation error, factor-subspace energy,
residual-subspace energy, horizon profiles, session effects, and the same
contrast in neuron space and PCA-10 space.

Overlapping test landmarks are dependent. Every ensemble uses the same historical
J5 innovations, and all contrasts are paired by session and landmark.

## Surrogates

The primary surrogate for the residual question keeps the observed factor
trajectory fixed, independently circular-shifts residual columns, and projects
the shifted residual back out of the factor subspace. This preserves the common
factor trajectory exactly while destroying residual cross-neuron alignment.

The complementary surrogate keeps residuals fixed and shifts factor scores; it
destroys common-factor temporal alignment while preserving individual residual
series. Full circular and 25-frame block shuffles are negative controls.

All surrogate draws are built only from the causal 500-frame window. A synthetic
test mutates frames after the landmark and verifies that the fitted operator is
unchanged.

## What the ladder can and cannot decide

A positive B3-B5 result would mean that the 10-factor control does not absorb all
prospective information in the population history. It would not identify direct
coupling, synapses, a command neuron, or a biological intervention target.

A null or negative B3-B5 result is a substantive negative finding: the earlier
B3 advantage can be explained by the matched own-history baseline and/or the
common factors. The audit will preserve that result rather than searching for a
different factor count.

## Dimensional correction

The earlier report's "103 predictors" is true only for session 0. The native
dimensions are 103, 127, 123, 118 and 121. All new models use the native session
dimension; the old reports remain unchanged.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
Biological intervention validity remains `NOT_EVALUABLE`.
