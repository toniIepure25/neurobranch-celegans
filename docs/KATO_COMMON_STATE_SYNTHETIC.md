# Kato common-state controls: synthetic audit

## Purpose

The synthetic systems are implementation diagnostics. They ask whether the
fixed control ladder can distinguish direct lagged coupling from a shared
external factor in a setting where the ground truth is known. They are not
biological evidence and do not validate the Kato result.

## Systems

Each system has 20 neurons, 2,400 frames, three seeds (11, 23, 37), a 1,600-frame
training block and a 1,200-frame causal fitting window. The five scenarios are:

1. **independent**: diagonal lag-one dynamics, no shared factor, no direct
   coupling;
2. **common**: a three-dimensional external factor with no direct
   cross-neuron coupling;
3. **coupled**: a sparse directed ring with no shared factor;
4. **coupled_common**: both the shared factor and the directed ring;
5. **time_varying_common**: a shared factor whose loadings rotate and change
   scale over time, with no direct coupling.

## Models and metric

B1m is a matched own-neuron AR. B3 is the full population VAR. B5-k is the
own-neuron-plus-factor model with `k=3` primary and `k=10` saturated diagnostic.
Forecasts are deterministic H20 conditional means from held-out landmarks.
The separation metric is `MSE(B5-k) - MSE(B3)`; positive means B3 is better.

## Decision rule

G5 passes only if the direct-coupling scenarios show a positive mean B3-B5
advantage and the common-only scenario does not. The time-varying common scenario
is descriptive: the fixed-factor control may fail there, and that failure is a
limitation of B5 rather than evidence for coupling.

## Results

Mean `MSE(B5-k) - MSE(B3)` over three seeds; positive means B3 is better:

| Scenario | k=3 | k=10 |
| --- | ---: | ---: |
| independent | -0.00031 | -0.00016 |
| common | -0.00068 | -0.00042 |
| coupled | +0.00219 | +0.00025 |
| coupled_common | +0.00182 | +0.00040 |
| time_varying_common | -0.00090 | -0.00051 |

All six direct-coupling systems (three seeds of `coupled` plus three of
`coupled_common`) have a positive k=3 advantage. The common-only scenario is
negative and the time-varying common scenario is negative. The saturated k=10
control shrinks the direct-coupling advantage but keeps its sign in both direct
scenarios. G5 PASS under the prespecified decision rule.

The `coupled_common` result is the important caveat: when a shared factor and
direct coupling coexist, the population advantage is small and the factor control
absorbs much of it. The Kato result should be read with the same caveat.

The generated table is `reports/simulator/kato_common_state/synthetic_audit.csv`,
with the scenario summary in `synthetic_summary.csv` and the machine-readable
verification in `synthetic_verification.json`.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
Biological intervention validity remains `NOT_EVALUABLE`.
