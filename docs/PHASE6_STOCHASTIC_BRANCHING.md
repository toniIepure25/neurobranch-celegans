# Phase 6 stochastic branching

## Question

Given a held-out reversal (REV) neural state, does the simulator reproduce the conditional distribution of the first behavioural outcome over 20 frames?

## Design

- Non-overlapping held-out REV states.
- Empirical reference: kNN over held-out REV states, excluding neighbours within `horizon+2=22` frames.
- Simulated: 80 native stochastic rollouts per state, matched initial history.
- Baselines: persistence readout, outcome marginal, first-order Markov chain fitted on training transitions.
- Metrics: total variation (TV), Jensen–Shannon divergence, entropy.
- No shuffled-null entropy inflation and no "bifurcation" language.

## Result

| Family | Mean TV vs observed | TV persistence | TV Markov | TV marginal | Sim entropy | Observed entropy |
|---|---:|---:|---:|---:|---:|---:|
| Stochastic VAR | 0.513 | 0.531 | **0.264** | **0.245** | 0.507 | 0.680 |
| Switching VAR | 0.657 | 0.531 | **0.264** | **0.245** | 0.711 | 0.680 |

Figure: `figures/simulator/phase6/branching_tv.png`.

### Interpretation

- Stochastic VAR is only marginally better than persistence (0.513 vs 0.531) and far worse than either the Markov chain or the marginal baseline.
- Switching VAR over-produces entropy (0.711 vs observed 0.680) while still having worse TV. High entropy is not evidence of realistic branching.
- The empirical conditional distributions are highly concentrated; the marginal and Markov baselines are strong. A simulator that produces diverse outcomes can be *less* accurate.
- No model demonstrates a branching advantage.

## Gate 4

**Fail.** The criterion required lower mean TV than both Markov and marginal baselines in at least 4/5 worms. The simulated TV was higher than both.

## Caveat

Behaviour labels are inferred from AVA/SMD/RIB calcium activity via a logistic probe. The probe is supported in >99% of states for stochastic VAR, so this is not primarily a readout-support failure; it is a mismatch in conditional outcome distributions.

## Files

- `reports/simulator/phase6/branching_summary.csv`
- `reports/simulator/phase6/branching_events.csv`
- `reports/simulator/phase6/branching_overview.json`
