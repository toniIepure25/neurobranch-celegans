# Phase 6 free-running validation

## Contract

- Real observed history window; no padding for the selected model.
- No teacher forcing.
- H=20: 50 test starts; H=100: 20 starts; H=200: 10 starts.
- Primary skill: \(1-\mathrm{MSE}/\mathrm{MSE}_{\text{constant initial state}}\).
- Test block 2 was inspected in Phase 5: **exploratory**, not confirmatory.

## Audit of Phase 5

The old code hardcoded `history=1` in `test_starts()`, padded `x0` for history>1, correlated concatenated blocks for its "autocorrelation" metric, and used one-step \(\Delta R^2\) for Gate 2. The corrected code reproduces stored h1 checkpoints exactly.

| Kind | Stored MSE | Recomputed MSE | Corrected autocorr error | Old autocorr error |
|---|---:|---:|---:|---:|
| VAR | 0.49913 | 0.49913 | 0.457 | 0.266 |
| selected MLP | 1.38802 | 1.38802 | 0.533 | 0.225 |
| GRU | 0.64075 | 0.66198 | 0.539 | 0.331 |

The GRU difference is the discarded two-frame history. The old autocorrelation metric understated mismatch in every family.

Figure: `figures/simulator/phase6/audit_autocorr.png`.

## Exploratory test results

Mean over five worms (three seeds for stochastic families).

| Family | H | Skill vs constant | Energy distance | Autocorr error | Covariance error | Readout support |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 20 | 0.000 | **0.203** | 0.289 | 0.395 | 0.968 |
| Phase 5 MLP | 20 | −1.356 | 1.957 | 0.276 | 1.655 | 0.718 |
| Phase 5 VAR | 20 | **0.101** | 2.078 | 0.275 | 0.635 | 0.998 |
| New VAR | 20 | 0.093 | 2.618 | 0.248 | 0.686 | 0.998 |
| Stochastic VAR | 20 | −0.410 | 1.277 | **0.169** | 0.656 | 0.998 |
| Switching VAR | 20 | −1.748 | 3.393 | 0.311 | 1.057 | 0.991 |
| Persistence | 100 | 0.000 | 0.535 | 0.416 | 0.680 | 1.000 |
| Phase 5 VAR | 100 | 0.254 | 3.573 | 0.346 | 0.781 | 1.000 |
| New VAR | 100 | **0.255** | 4.630 | 0.289 | 0.858 | 1.000 |

Figure: `figures/simulator/phase6/test_skill.png`.

### What the numbers mean

- The selected new VAR beats persistence in 4/5 worms at H=20, but **loses to the frozen Phase 5 VAR** (mean +0.093 vs +0.101).
- The Phase 5 residual MLP, previously selected by validation one-step \(R^2\), is catastrophically worse than persistence at H=20 and H=100. Its behaviour readout is supported in only 21.5% of states at H=100.
- Stochastic models have better autocorrelation than persistence but much worse energy distance and covariance error. They do not reproduce the joint distribution.
- No model has lower energy distance than persistence. The constant-state baseline is the best distributional model in this comparison.
- All long H=200 rollouts were finite, but finite does not mean accurate or supported.

## Gate 2

**Fail.** The criterion required a mean skill >0.02, a positive block-bootstrap lower CI, and superiority to both persistence and the frozen Phase 5 VAR in at least 4/5 worms. The selected VAR meets the persistence comparison but not the frozen-VAR comparison.

## Gate 3

**Fail.** A model needed lower energy, autocorrelation and covariance error than persistence in at least 4/5 worms. No family did; stochastic VAR improved only autocorrelation.

## Bottom line

The simulator runs autonomously and stably, but it is not a validated generative model of the validation or test distributions. The correct label is **negative/limited result**, not "validated digital twin".
