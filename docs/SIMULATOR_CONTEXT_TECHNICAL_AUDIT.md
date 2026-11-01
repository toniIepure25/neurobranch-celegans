# Technical audit — context-conditioned dynamics

## 1. Starting state

- Branch: `research/context-conditioned-generative-dynamics`
- Base commit: `4da6c24236d2c6399f5f1b859a21ae479e3521cc`
- Frozen Phase 6 reports under `reports/simulator/phase6/` were not rewritten, relabelled, or used as a source of new checkpoints.
- No raw datasets were added to Git.

## 2. What was corrected, not hidden

### 2.1 VAR stability

The legacy path scaled the VAR coefficient matrix once. That does not guarantee that the *actual* companion matrix has the requested radius. `src/neurobranch/dynamics/stability.py` now:

1. builds the companion matrix explicitly;
2. computes its spectral radius;
3. iteratively scales only the AR coefficient rows while preserving the identity block;
4. verifies the realized radius after every update.

`StableVARModel` is a new class (`var_stable`). It does not mutate a frozen checkpoint. Its `stability_record()` stores the requested cap, realized radius, iteration count, and final scale.

### 2.2 Hashing

`src/neurobranch/reproduction/hashes.py` separates:

- **raw byte** SHA256 for binaries and exact files;
- **canonical text** SHA256 after UTF-8 decoding and CRLF/CR → LF normalisation.

Manifest records state `algorithm` and `normalization`. A legacy bare digest is accepted only through an explicit compatibility path. This addresses the Phase 6 observation that 6 artifacts matched raw bytes and 40 matched only after LF normalisation. The frozen Phase 6 manifest was not rewritten.

### 2.3 Support audit

`SupportModel` now retains:

- calibration scores;
- q90/q95/q99 thresholds;
- raw distances;
- `threshold_at(q)` and `support_table(X)`.

Phase 6's official q99 threshold is unchanged. The new audit in `reports/simulator/context_dynamics/` reports q90/q95/q99 as sensitivity analyses. It also records that Phase 6's per-start table stores only a rollout's maximum distance, so the audit is conservative and cannot reconstruct a full per-step distance distribution without re-simulation.

Important observed deviation: at q99, the Phase 6 `switching_var` family has only **0.86** mean fraction of starts whose maximum distance remains inside the q99 threshold. This is a support diagnostic, not a retroactive change to the frozen result.

### 2.4 Latent-model causality

The latent LDS uses:

- PCA fitted on training frames only;
- a ridge latent transition fitted on training transitions only;
- an initial covariance from training data only;
- a causal filter that never receives a future observation;
- a coefficient policy that receives only the current encoded state and current known input.

Tests compare a short filter prefix with the same prefix of a longer filter and require exact agreement.

### 2.5 External oxygen alignment

`build_kato_stim_context_worm_data` attaches the **full-recording** measured oxygen series. It does not assume that metadata switch times land exactly on a 30 s frame grid. The audit records actual transition frame times; e.g. the first transition is 360.000 s in some sessions and 360.178 s in others. A test checks alignment to the recording's actual `time_seconds`.

### 2.6 Coefficient trajectories

The first draft used a full-recording Kalman pass to produce operator coefficients. That was replaced by a causal encoded-state trajectory (`causal_encoded_state`), which uses only the current observation and never a future coefficient or future observation. The replacement is faster and has a clearer causal interpretation; it is labelled differently from a filtered posterior.

## 3. New code

| Path | Purpose |
|---|---|
| `src/neurobranch/dynamics/stability.py` | companion matrix, exact iterative cap |
| `src/neurobranch/dynamics/stable_var.py` | corrected stable VAR |
| `src/neurobranch/dynamics/context_models.py` | stable latent LDS, compositional LDS, shared latent model |
| `src/neurobranch/reproduction/hashes.py` | raw/canonical hashing |
| `src/neurobranch/data/external.py` | context oxygen adapter |
| `src/neurobranch/simulator/adapters.py` | q90/q95/q99 support, known schedule input |
| `src/neurobranch/simulator/metrics.py` | support audit fields, external-input rollout |
| `src/neurobranch/dynamics/checkpointing.py` | serialization of new model kinds |
| `app/context_ui.py` | “CONTEXT-AWARE NEURAL SIMULATOR” tab |

## 4. Test coverage added

- exact companion cap and realized-radius verification;
- stable VAR round trip and radius check;
- raw vs canonical hashing and manifest verification;
- train-only PCA, no-future filtering, causal external input;
- compositional simplex coefficients, stable operators, no oracle coefficients;
- support q90/q95/q99 raw-distance API;
- external oxygen alignment and masks;
- bounded perturbation amplitudes;
- serialization of latent, compositional, stable-VAR, and shared models;
- dashboard safety-label and missing-report contract.

Tests are run with `python -m pytest`; Ruff and the full suite results are recorded in `_RESULTS.md`.

## 5. Known limitations of the implementation

- The shared-latent model pools independently fitted PCA coordinates. Shared `A` is a model coordinate, not a biological common coordinate. The calibrated map `R` is fitted on a calibration prefix, but it cannot repair a fundamentally non-identifiable basis.
- Compositional operators are small and fitted by alternating NNLS/ridge. Coefficient collapse is a possible outcome and is reported, not hidden.
- The support diagnostic uses nearest-neighbour distance in standardized neural space. It is not a likelihood and not a mechanistic boundary.
- The oxygen schedule is measured environmental input, not targeted neuronal stimulation. Session count and transition count are small.
- The test block for NoStim is a new chronological split but is not independent of the recordings used for model development in earlier phases.
