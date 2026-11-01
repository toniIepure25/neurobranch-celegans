# Branching and alternative futures

## 1. Method

For each held-out reversal state, the selected latent LDS is rolled out with stochastic draws from its **fitted latent process covariance**. No arbitrary observation-space noise is added. This is model uncertainty, not measured biological variability.

Comparators:

- persistence;
- first-order Markov chain fitted on training macrostate transitions;
- marginal first-exit distribution;
- observed local kNN outcome distribution around the same state.

Metrics: entropy, total variation, Jensen–Shannon divergence, OOD fraction, number of observed neighbours, and number of rollouts.

## 2. What is not claimed

- No claim that the living network contains a bifurcation.
- No claim that the model has discovered a biological branch point.
- Agreement with the empirical local distribution is a diagnostic, not validation.
- The process covariance is estimated from training residuals and may be misspecified.

## 3. Results

Four of five sessions had evaluable reversal states; session 1 had no REV candidate frames.

| Comparator | Mean TV to observed local outcome |
|---|---:|
| fitted stochastic latent LDS | 0.580 |
| persistence | 0.548 |
| marginal first-exit | 0.317 |
| first-order Markov | 0.360 |

The latent ensemble was worse than persistence and much worse than Markov/marginal. Its mean simulated entropy (0.520) was below observed local entropy (0.652). This is a misspecified fitted-uncertainty model, not evidence of a biological bifurcation.

## 4. Reproduction

```powershell
python scripts/context_branching.py --n-rollouts 60
```
