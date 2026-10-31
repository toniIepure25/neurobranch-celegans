# Phase 6 final scientific audit

**Audit date:** 2026-10-31. **Official Phase 6 head:** `9d54381`.
This audit is read-only with respect to the official Phase 6 reports. It does
not retroactively change the 99th-percentile result or any gate.

## What Phase 6 asked

> Can a fitted generative model produce autonomous *C. elegans* neural
> trajectories from real observed histories, reproduce multi-horizon dynamical
> and distributional statistics, and support bounded named-neuron perturbation
> analysis without extreme extrapolation?

The primary object is the neural state. Behaviour decoding is a supporting
readout. The simulator contract is free-running:
`x(t+1) = F_theta(history, u) + epsilon`, with no teacher forcing during
rollout.

## Frozen design

- Data: Kato 2015 `WT_NoStim`, five worms, 1080 s, 3 Hz, 109–135 traces.
- Split: five blocked segments; validation block 4 used for selection; test block
  2. Test block 2 had already been inspected in Phase 5.
- Selection metric: validation free-running skill at H=20 versus persistence.
- Primary model families: VAR, Gaussian stochastic VAR, switching VAR with K=3.
- Primary metric: `S = 1 - MSE_model / MSE_persistence` at H=20.
- Support: kNN support calibrated on validation distances; official threshold
  the 99th percentile; 90% and 95% reported as sensitivity.
- Perturbations: observed one-step amplitude capped to the central 90% marginal
  range, support-gated; outputs are model-dependent, not biological.

The selected configurations and the frozen selection table are in
`reports/simulator/phase6/selection_frozen.json`. No B4 family was pursued
because no B1–B3 configuration had positive validation skill in at least 3/5
worms.

## Validation selection

| Family | Selected configuration | Validation median skill | Positive worms |
| --- | --- | ---: | ---: |
| VAR | lag 2, ridge 10, cap 0.995 | −0.2209 | 0/5 |
| Stochastic VAR | Gaussian, same lag/ridge | −0.7820 | 0/5 |
| Switching VAR | K=3 | −3.3084 | 0/5 |

No learned family had positive median validation skill. The VAR was selected as
the least-negative family, not as a validated simulator.

## Exploratory test results

Because test block 2 was reused, these are exploratory and cannot be called
confirmatory.

| Family | H=20 mean skill | H=100 mean skill |
| --- | ---: | ---: |
| Persistence | 0.000 | 0.000 |
| Frozen Phase 5 VAR | +0.1008 | +0.2544 |
| Phase 6 VAR | +0.0930 | +0.2549 |
| Stochastic VAR | −0.4096 | −0.1730 |
| Switching VAR | −1.7478 | −0.7209 |
| Phase 5 MLP | −1.3556 | −17.999 |

The Phase 6 VAR is not better than the frozen Phase 5 VAR. Its H=20 confidence
interval is positive but small (0.0028–0.1868), and the gate required both a
meaningful effect and superiority to the frozen baseline.

## Gates

| Gate | Question | Status | Audit note |
| --- | --- | --- | --- |
| G1 | Valid finite free-running rollout | pass | 5/5 finite |
| G2 | Autonomous skill and baseline superiority | **fail** | 4/5 vs persistence, only 3/5 vs frozen Phase 5 VAR; below criterion |
| G3 | Dynamical/distributional fidelity | **fail** | 0/5 improve energy, autocorrelation and covariance simultaneously |
| G4 | Stochastic branching fidelity | **fail** | TV 0.513 vs Markov 0.264 and marginal 0.245 |
| G5 | Support-constrained perturbation | pass at q=0.99 | 36/36 conditions in support; threshold-sensitive, see below |
| G6 | Robust named-neuron sensitivity | **not_evaluable** | only one model family; protocol requires ≥2/3 |
| G7 | Biological validation | **not_evaluable** | Kato NoStim has no matched intervention |

Negative conclusions preserved:

- No learned family beat persistence by median validation skill.
- The selected VAR's exploratory test advantage is small and not superior to the
  frozen Phase 5 VAR.
- Distributional and branching fidelity fail.
- There is no robust, multi-family named-neuron effect.
- There is no biological or causal validation.
- The model must not be called a digital twin or a validated causal simulator.

## The 95th-to-99th-percentile support deviation

The original protocol named a 95th-percentile primary support threshold. The
executed default was the 99th percentile, calibrated on validation distances.
The deviation was documented before test evaluation in
`docs/PHASE6_SIMULATOR_DEVIATIONS.md`; the protocol wording was later aligned
with the executed 99th-percentile default. That does **not** make the 95th
percentile the executed result. The official result remains q=0.99.

### What the audit recomputed

From `support_sensitivity.csv` (90/95/99):

- q95/q99 threshold ratio ranges from 0.742 to 0.963 across worms.
- Observed test support at 90/95/99 is 92.9/94.0/97.4% overall.
- Selected VAR and stochastic VAR rollouts are 100% in support at all three
  quantiles.
- Switching VAR rollout support is 91.7/93.7/99.1%; at q=0.95 it falls to 0.755
  for worm 3 and 0.93 for worm 4.
- For worm 3, observed test support falls from 91.7% at q=0.99 to 78.4% at
  q=0.95; worm 4 falls from 97.7% to 93.9%.

### What cannot be reconstructed

`perturbation_events.csv` stores in-support fractions at q=0.99 but not support
distances or thresholds. Condition-level G5 at q=0.95 cannot be exactly
recomputed from the published event table. The rollout evidence makes a q=0.95
failure plausible, especially for switching VAR, but it is not a condition-level
recomputation. A bounded rerun would require the checkpoints, the same starts and
seeds, and a separate audit output; it was not performed here.

**Conclusion:** G5 is a q=0.99 pass and a threshold-sensitive result. It is not
evidence that the perturbation interface is robust at the originally named 95th
percentile. The official q=0.99 result is preserved; the q=0.95 status is
`not_evaluable_at_condition_level`.

## Additional audit findings

### Nominal spectral cap is not exactly realized

`VARModel.fit` applies a one-shot scaling of the coefficient matrix when the
companion spectral radius exceeds the cap. Because the companion matrix retains
an identity block, scaling the coefficient block does not scale all eigenvalues
linearly. The realized radii of the committed VAR and stochastic VAR checkpoints
are all slightly above the nominal 0.995:

| Worm | Realized radius |
| ---: | ---: |
| 0 | 0.9960 |
| 1 | 0.9960 |
| 2 | 1.0189 |
| 3 | 0.9963 |
| 4 | 1.0498 |

This is an implementation limitation of the frozen pipeline, not a post-hoc
reason to change the official result. It is relevant to the interpretation of
long-horizon stability and to any future refit. The external pilot reports its
own realized radii and does not apply a post-hoc fix.

### Manifest hashes are LF-normalized, not raw-byte stable on Windows

`simulator_manifest.json` lists 46 artifacts. Verification in the Windows
working tree gives:

- 6 raw-byte matches;
- 40 matches only after normalizing CRLF to LF;
- 0 other mismatches;
- 0 missing files.

All 46 artifacts therefore verify under the manifest's LF-normalized convention.
A naive raw-byte verifier reports 40 failures. This is a provenance/tooling
defect, not evidence that the scientific reports were altered. The audit writes
both counts rather than silently rewriting the official manifest.

### Test-block reuse remains the dominant inferential limitation

Phase 5 inspected test block 2. Phase 6 did not have a fresh confirmatory test
set. No amount of gate arithmetic repairs that. The correct description is
“exploratory Phase 6 test evidence”, not “confirmatory validation”.

## External consequence

The external readiness audit finds:

- frozen transfer to Kato Stim or Nichols is blocked by feature-space and
  identity mismatch;
- Kato Stim supports a limited per-animal B pilot on the constant-21%-oxygen
  first 360 s;
- Nichols is not a like-for-like cohort and was not executed;
- oxygen is environmental input, not targeted neuronal stimulation.

The Kato Stim pilot is reported separately in
[EXTERNAL_DATASET_COMPATIBILITY.md](EXTERNAL_DATASET_COMPATIBILITY.md) and
`reports/external_validation/kato_stim_pilot_summary.json` (readable summary:
`reports/external_validation/kato_stim_pilot.md`). Its result is negative (0/7
animals beat persistence), which is consistent with the Phase 6 negative
conclusion and does not rescue the model.

## What would be needed for a stronger claim

- A fresh, untouched cohort with documented animal independence.
- A common neuron mapping or a principled latent alignment model.
- A pre-registered model and threshold, fixed before any target test inspection.
- An input-aware model with randomized, controlled sensory input if the question
  is input response.
- Independent behavioural measurement if the question is behaviour.
- A matched intervention if the question is causal.

None of these is established by the current data.

## Reproduction

- `python scripts/phase6_threshold_audit.py`
- `reports/external_validation/phase6_threshold_audit.json`
- `reports/external_validation/phase6_freeze.json`
- `reports/external_validation/phase6_freeze_manifest.sha256`
- Official reports remain under `reports/simulator/phase6/`.
