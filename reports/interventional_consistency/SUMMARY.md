# Stage One — interventional consistency of neural abstractions

Protocol: `37eba808bc549a39…` (`ic1`, frozen).
Seed: `0`. Gates passed: **9/9**.

> Synthetic ground truth only. Nothing below is evidence about *C. elegans*, and no
> external method was reproduced (see `CONFORMANCE.md`).

## Gates

| gate | result | detail |
|---|---|---|
| G1a_independent_reference | PASS | exact_commutation completed with DiscreteFixture.simulate disabled |
| G1b_simulation_agreement | PASS | the sampling estimator reproduces the enumerator within its declared noise floor on 4/4 operator-fixture pairs |
| G2_positive_control | PASS | the exactly lumpable fixture has zero dynamic error with at least one genuine pair per cell |
| G3_negative_controls | PASS | behaviour-only and intervention-only negatives separate on the intended estimand and only there |
| G3b_history_ladder | PASS | history length 1 fails and length 2 succeeds, exactly |
| G4_contract_enforcement | PASS | 'pca_observation' and 'oracle' cannot be compared as representations: pca_observation vs oracle: NOT comparable as representations (information differs: behavior_labels False->True, true_latent_state False->True) |
| G5_identifiability_witness | PASS | the witness operator's error stays above tolerance and does not shrink with the window |
| G6_determinism | PASS | two in-process runs with the same seed produced identical metrics |
| G7_frozen_history | PASS | 44 recorded historical manifests re-hashed |

## Exact commutation errors

| fixture | operator | behaviour | dynamic | worst case | cells | singleton | unsupported |
|---|---|---|---|---|---|---|---|
| exactly_lumpable/positive | passive | 0.0000 | 0.0000 | 0.0000 | 3 | 0 | 0 |
| exactly_lumpable/behaviour_only | passive | 0.0000 | 0.1800 | 0.1800 | 3 | 0 | 0 |
| exactly_lumpable/intervention_only | passive | 0.0000 | 0.0000 | 0.0000 | 3 | 0 | 0 |
| exactly_lumpable/intervention_only | lesion | 0.0000 | 0.5000 | 0.5000 | 3 | 0 | 0 |
| ring/L1 | passive | 0.0000 | 0.7000 | 0.7000 | 3 | 0 | 0 |
| ring_witness/L1 | passive | 0.0000 | 0.7000 | 0.7000 | 3 | 0 | 0 |
| ring_witness/L1 | asym | 0.0000 | 0.8500 | 0.8500 | 3 | 0 | 0 |
| ring/L2 | passive | 0.0000 | 0.0000 | 0.0000 | 6 | 6 | 3 |
| ring_witness/L2 | passive | 0.0000 | 0.0000 | 0.0000 | 6 | 0 | 3 |
| ring_witness/L2 | asym | 0.0000 | 0.8500 | 0.8500 | 6 | 0 | 3 |
| ring/L4 | passive | 0.0000 | 0.0000 | 0.0000 | 24 | 24 | 57 |
| ring_witness/L4 | passive | 0.0000 | 0.0000 | 0.0000 | 24 | 0 | 57 |
| ring_witness/L4 | asym | 0.0000 | 0.8500 | 0.8500 | 24 | 0 | 57 |

## Sampling versus enumeration

| pair | exact | estimated | gap | noise floor |
|---|---|---|---|---|
| exactly_lumpable/positive|passive | 0.0000 | 0.0115 | 0.0115 | 0.0500 |
| exactly_lumpable/behaviour_only|passive | 0.1800 | 0.1859 | 0.0059 | 0.0500 |
| exactly_lumpable/intervention_only|passive | 0.0000 | 0.0115 | 0.0115 | 0.0500 |
| exactly_lumpable/intervention_only|lesion | 0.5000 | 0.4958 | 0.0042 | 0.0500 |

## Continuous system (descriptive, no gate)

| operator | worst case | mean pairwise | behaviour | cells |
|---|---|---|---|---|
| passive | 1.000 | 0.512 | 0.300 | 16 |
| lesion | 1.000 | 0.392 | 0.387 | 16 |
| clamp | 1.000 | 0.289 | 0.200 | 16 |

The worst-case error saturates at 1 for a hard partition of a continuous state:
members at opposite edges of a cell leave through different faces. The mean pairwise
term is the graded quantity there. Note that it is *lower* under two of the operators
than under the passive dynamics, which is the substantive point — commutation is a
property of the pair (abstraction, operator), not of the abstraction alone.

## Verdict vocabulary

| label | meaning here |
|---|---|
| `verified` | checked against a source in this environment |
| `simulated` | our own synthetic ground truth |
| `unidentifiable` | cannot be estimated from the permitted information |
| `falsified` | tested and failed |
| `not_reproduced` | the external implementation did not run here |
