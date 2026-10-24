# Phase 4 independent replication of the Phase 3 prospective benchmark

## Purpose

Phase 3 reported a 6 s behaviour log loss of 0.5927 and a neural-augmented
0.5856, a paired difference of −0.00701 with 95% CI
[−0.01483, +0.00053].  That primary criterion already failed because the CI
included zero.  Phase 4 repeats the **same analysis protocol** on an
independent WT octanol cohort from Kramer et al. (2026).

This is a replication of the protocol, not of the exact recordings.  It is
also not a replication of the paper's turn-direction RNN.

## Protocol

- 32 WT octanol animals; 48,877 frames; 990 reversals.
- Phase 3 feature builder and target: reversal active at `t + h`, eligible
  frames exclude the reversal window, 10-frame history.
- Horizons: 3, 6 and 12 s.
- Modes: constant, state, behaviour, neural, behaviour+neural,
  behaviour+neural PCA(10), delayed-neural control, quadratic behaviour,
  quadratic+neural, behaviour+identified.
- Fixed `StandardScaler` + logistic regression, `C=0.1`, `lbfgs`,
  leave-one-animal-out.
- Uncertainty: 10,000 animal-level bootstrap resamples of per-animal mean
  log-loss differences.
- Primary criterion (prespecified in Phase 3): mean Δ < 0, 95% CI upper < 0,
  and at least 70% of animals improved.

## Results

| horizon | behaviour log loss | +neural log loss | Δ | 95% CI | animals improved | criterion |
|---:|---:|---:|---:|---|---:|---|
| 3 s | 0.3634 | 0.3597 | −0.00372 | [−0.00746, +0.00018] | 22/32 (68.75%) | not met |
| 6 s | 0.5452 | 0.5395 | −0.00570 | [−0.01133, +0.00047] | 22/32 (68.75%) | not met |
| 12 s | 0.6368 | 0.6323 | −0.00455 | [−0.01304, +0.00496] | 23/32 (71.9%) | not met |

The direction and magnitude at 6 s are consistent with Phase 3
(−0.00570 vs −0.00701), but the CI still includes zero and the fraction of
improved animals is below the 70% threshold.  The 3 s result is also null.
The 12 s result is null.

Secondary modes at 6 s:

| contrast | Δ log loss | 95% CI | interpretation |
|---|---:|---|---|
| neural alone vs behaviour | +0.0517 | [+0.0382, +0.0654] | worse |
| behaviour + PCA(10) | −0.00354 | [−0.00738, +0.00022] | null |
| behaviour + identified | −0.01212 | [−0.02144, −0.00153] | worse |
| shifted-neural control | −0.00799 | [−0.01412, −0.00134] | control also “improves” |
| neural within quadratic behaviour | −0.00276 | [−0.00978, +0.00639] | null |

The shifted-neural control is important: simply shifting neural features also
produces a small negative log-loss difference, so the tiny neural effect is
not specific to the contemporaneous neural window.

## Deviations from Phase 3

- Different cohort: 32 Kramer WT octanol animals, not the 27 Phase 3 animals.
- `distance_from_center` is a centroid-distance proxy because the Baskoylu
  copper-centre coordinate is absent.
- Angular velocity is read from the HDF5 behaviour group rather than derived
  with the Phase 3 formula.
- `head_angle` is the orientation-corrected `dv`; raw camera sign is not used.
- No hyperparameter tuning on the new cohort; `C=0.1` is fixed.

These deviations are sufficient to make this an independent replication
rather than an exact reanalysis.  They are not sufficient to explain the
consistent null, because the effect direction is reproduced but the
uncertainty and animal-improvement criteria are not.

## Conclusion

Phase 4 **does not replicate** the Phase 3 primary neural-increment claim.  It
reproduces the small negative point estimate, but not a CI excluding zero or
the 70% animal-improvement threshold.  The correct description is a
non-replication of a small, uncertain effect, not evidence that neural
activity is irrelevant.

## Outputs

- `reports/phase4/replication_metrics.csv`
- `reports/phase4/replication_deltas.csv`
- `reports/phase4/replication_summary.json`
- `reports/phase4/replication_protocol.json`
- `reports/phase4/figures/phase4_replication_delta.png`
