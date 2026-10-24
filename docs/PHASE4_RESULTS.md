# Phase 4 results: turn-direction generalization

## One-paragraph answer

Phase 4 tested whether the missing neural signal in Phase 3 was due to an
inappropriate target, insufficient representations, distribution shift, or a
genuine absence of prospectively decodable information.  The independent 6 s
replication did **not** meet its prespecified criterion (Δ log loss −0.0057,
95% CI [−0.0113, +0.0005], 22/32 animals improved).  A new animal-disjoint
dorsal/ventral turn-direction benchmark found a small positive point estimate
for population activity (Δ AUROC +0.0123, 95% CI [−0.0166, +0.0425]) but no
uncertainty-bounded increment.  SAAV, circuit and identified-neuron
representations were null or worse; nonlinear boosting did not rescue the
effect; cohort, window and external-transfer analyses were not consistently
positive.  The defensible conclusion is **no evidence of incremental
prospective turn-direction information in this archive under animal-disjoint
evaluation**, not proof that the circuit is uninvolved.

## Repository state

- Project: `D:\ComputaCenter\NeuroBranch\neurobranch-celegans`
- Phase 3 branch: `research/phase3-neural-generalization`
- Phase 3 base commit: `66cf7bbd2aa21887ad5716e6bae1e0768eaa1c86`
- Phase 4 branch: `research/phase4-turn-direction-generalization`
- No remote push, force-push or history rewrite was performed.
- Phase 1–3 code, data, reports, warnings and history are preserved.

## Data and provenance

- Dryad `doi:10.5061/dryad.8sf7m0cz2`, version 5, dataset id 135272,
  version id 433175, CC0-1.0, 82 files / 1,455,335,690 bytes.
- 80 files downloaded (1,444,081,660 bytes); all SHA-256 checksums match the
  Dryad manifest.  The two omitted files are MATLAB `.chxTracks.mat` behaviour
  tracking files, not used here.
- 73 recordings, 107,131 frames, 2,670 reversals, 7,025 labelled traces, 115
  identified labels.
- WT octanol: 32 animals, 48,877 frames, 990 reversals, 596 long valid events.
- WT diacetyl: 24 animals, 34,490 frames, 1,013 reversals, 349 long valid.
- tdc1 octanol: 17 animals, 23,764 frames, 667 reversals, 282 long valid.
- Primary WT turn-direction set: 56 animals, 945 events, 186 dorsal / 759
  ventral (19.7% dorsal).
- The archive has no explicit D/V labels.  `dv` is the orientation-corrected
  D/V angle (positive = dorsal); raw `head_angle` has a per-recording camera
  sign and is not used across animals.  Marker-neuron pre-offset activity is
  consistent with the sign convention, but this is a sanity check, not proof.
- No trial-level single-cell optogenetic recordings are present.

## Independent 6 s replication

Same Phase 3 protocol, fixed `C=0.1`, LOAO, 32 WT octanol animals:

| horizon | behaviour | +neural | Δ | 95% CI | improved | criterion |
|---:|---:|---:|---:|---|---:|---|
| 3 s | 0.3634 | 0.3597 | −0.00372 | [−0.00746, +0.00018] | 22/32 | not met |
| **6 s** | **0.5452** | **0.5395** | **−0.00570** | **[−0.01133, +0.00047]** | **22/32** | **not met** |
| 12 s | 0.6368 | 0.6323 | −0.00455 | [−0.01304, +0.00496] | 23/32 | not met |

The 6 s point estimate is close to Phase 3 (−0.00701), but the CI includes
zero and the fraction improved is 68.75%, below the 70% criterion.  This is a
non-replication, not a positive result.  A shifted-neural control also shows a
small negative difference (−0.0080, [−0.0141, −0.0013]), so the tiny effect is
not specific to the contemporaneous neural window.

## Turn-direction benchmark

Primary model: StandardScaler + logistic regression, `C=0.1`, LOAO.  The
primary feature segment is four middle frames of a completed reversal with
`seg_end <= end - 2`: a two-frame guard band separates the last feature frame
from the reversal offset.  No post-guard frame is a feature.  Segment location
is defined relative to the completed reversal, as in the paper's offline task;
this is not an online onset-triggered decoder.

| mode | AUROC | AUPRC | log loss | Brier | balanced accuracy | calibration slope |
|---|---:|---:|---:|---:|---:|---:|
| behaviour | 0.595 | 0.259 | 0.4987 | 0.1566 | 0.499 | 0.428 |
| behaviour + population | **0.608** | **0.293** | **0.4971** | **0.1552** | 0.510 | 0.470 |
| behaviour + SAAV | 0.591 | 0.258 | 0.5000 | 0.1569 | 0.499 | 0.404 |
| behaviour + circuit | 0.586 | 0.263 | 0.5064 | 0.1580 | 0.504 | 0.318 |
| behaviour + identified | 0.577 | 0.272 | 0.5848 | 0.1777 | 0.537 | 0.176 |
| neural population only | 0.556 | 0.244 | 0.4963 | 0.1578 | 0.500 | 0.480 |
| neural circuit only | 0.503 | 0.207 | 0.5038 | 0.1606 | 0.500 | 0.020 |
| neural SAAV only | 0.379 | 0.154 | 0.4994 | 0.1591 | 0.500 | −4.94 |

Primary contrast: **Δ AUROC +0.0123**, 95% animal-cluster bootstrap CI
**[−0.0166, +0.0425]**, `p(Δ≤0)=0.210`; classification
**no incremental evidence**.  Per-animal AUROC improves in 25/50 animals,
worsens in 20/50, ties in 5/50; median Δ +0.017, mean +0.006.  Calibration is
poor (slope 0.43–0.47).  The event metrics and per-event parquet are retained
so the result can be re-analysed without refitting.

Prespecified neural contrasts:

- SAAV: Δ −0.0042, CI [−0.0117, +0.0029].
- Circuit: Δ −0.0097, CI [−0.0380, +0.0191].
- Identified: Δ −0.0176, CI [−0.0991, +0.0574].

Cohort/window sensitivity:

| analysis | behaviour | +population | Δ | 95% CI |
|---|---:|---:|---:|---|
| WT octanol | 0.596 | 0.596 | −0.0002 | [−0.0429, +0.0394] |
| WT diacetyl | 0.519 | 0.582 | +0.0624 | [−0.0040, +0.1241] |
| early window | 0.593 | 0.569 | −0.0243 | [−0.0520, +0.0009] |
| late window | 0.618 | 0.619 | +0.0013 | [−0.0322, +0.0348] |

The diacetyl result is secondary, exploratory, based on 349 events, has a
near-chance behaviour baseline, and its CI includes zero.  It is not the
headline.  The earlier feature set included completed-reversal duration and
time-to-offset; those are future information and were removed before this
final run, after which the diacetyl point estimate attenuated from a
CI-excluding-zero exploratory result to the null-compatible estimate above.

SAAV-available subset (27 animals, 445 events): behaviour 0.475, +SAAV 0.479,
SAAV-only 0.482; Δ +0.0045, CI [−0.0127, +0.0210].  No evidence.

External transfer (descriptive, no formal CIs):

| train → test | behaviour | +population | +SAAV | +circuit |
|---|---:|---:|---:|---:|
| octanol → diacetyl | 0.564 | 0.551 | 0.565 | 0.571 |
| WT → tdc1 | 0.621 | 0.614 | 0.629 | 0.609 |
| octanol → tdc1 | 0.612 | 0.626 | 0.621 | 0.576 |

No robust cross-cohort neural gain.  The octanol → diacetyl transfer does not
reproduce the within-diacetyl exploratory result, consistent with noise or
cohort-specific effects.

Nonlinear sensitivity: HistGradientBoosting behaviour AUROC 0.605,
behaviour + population 0.617; Δ +0.0121, CI [−0.0275, +0.0481].  A stronger
nonlinear behaviour model still shows no bounded neural increment.

Within-animal five-fold comparison (non-primary): behaviour 0.584,
+population 0.602, +circuit 0.577.  This is a comparison, not an
animal-disjoint result; it does not rescue the primary null.

## Neural dynamics

Time-resolved single-frame LOAO:

| frame | behaviour | neural | combined |
|---:|---:|---:|---:|
| −12 | 0.578 | 0.541 | 0.591 |
| −4 | 0.616 | 0.544 | 0.616 |
| −2 | 0.508 | 0.561 | 0.556 |
| −1 | 0.737 | 0.539 | 0.738 |
| 0 | 0.978 | 0.509 | 0.978 |
| +2 | 0.886 | 0.522 | 0.885 |

The outcome is nearly determined by behaviour at the offset-adjacent frame,
which is expected autocorrelation and is excluded from the primary window.
Once that frame is excluded, neural activity does not add a stable increment.
PCA of the population state explains 60.7/32.7/4.1% in the first three
components and shows a descriptive dorsal/ventral trajectory separation.
Marginal peak times differ by direction for SMDD, SAAV, RIV and SMDV, but the
SDs are 5–9 frames and the analysis is descriptive, not a fitted sequence.

## Controls and intervention evidence

- Shifted-neural control: population Δ −0.0164; SAAV +0.0061; circuit −0.0011
  on 913 common events.
- Circuit identity permutation: 0.5824 vs 0.5857, Δ −0.0033.
- Within-animal label permutation (6 draws): observed Δ +0.0125; null mean
  −0.0005, SD 0.0226; one-sided empirical p = 0.333.
- Adjacent reversal overlap: 0/945.
- Matched random-neuron ablation (24 random draws): every prespecified
  single-feature drop lies inside the random-control distribution
  (Δ AUROC −0.0040 to +0.0050; random mean +0.0012, SD 0.0033; all empirical
  p ≥ 0.12).  No neuron-specific feature dependence is established.

No trial-level optogenetic data are available.  The tdc1 cohort is treated as
a distribution-shift cohort, not an intervention.  No virtual perturbation is
presented as causal.  Any model-based perturbation display must carry:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Engineering and validation

- New adapter: `src/neurobranch/data/kramer.py` (raw HDF5 traces, orientation,
  event validation, Phase 3 wrapper).
- New benchmark: `src/neurobranch/behavior/turn_direction.py` (designs,
  matrices, LOAO, metrics, calibration, within-animal and external evaluation,
  ablations).  Behaviour features exclude reversal duration and time-to-offset;
  odour context is truncated at the feature segment end.
- Scripts: `scripts/phase4_download_kramer.py`, `phase4_audit_data.py`,
  `phase4_independent_replication.py`, `phase4_turn_direction.py`,
  `phase4_dynamics.py`, `phase4_controls.py`, `phase4_ablation.py`,
  `phase4_nonlinear.py`, `phase4_make_figures.py`,
  `phase4_write_manifest.py`.
- Tests: `tests/data/test_kramer.py` covers event/segment guards, orientation
  and prospective-window rules.
- Dashboard: 9th tab, “Turn Direction & Neural Sequences”, reads the saved
  reports and displays the null primary result, calibration, cohort
  heterogeneity, dynamics and the intervention warning.
- Raw data remain untracked; only code, docs, compact reports and figures are
  committed.

## Figures

- `reports/phase4/figures/phase4_turn_direction_auroc.png`
- `phase4_turn_direction_delta_forest.png`
- `phase4_turn_direction_calibration.png`
- `phase4_turn_dynamics_timeresolved.png`
- `phase4_replication_delta.png`
- `phase4_turn_pca_trajectories.png`
- `phase4_turn_neuron_peak_times.png`
- `phase4_turn_direction_external.png`
- `phase4_event_inventory.png`
- `phase4_turn_neuron_ablation.png`
- `phase4_turn_direction_nonlinear.png`

## Interpretation by hypothesis

- **Inappropriate target:** changing to dorsal/ventral direction did not
  reveal a neural increment in the primary cohort.  Target choice alone is not
  sufficient.
- **Insufficient representation:** population, SAAV, circuit, identified and
  nonlinear models all failed to produce a bounded positive increment.  This
  weakens, but does not eliminate, a representation/alignment explanation;
  the paper's exact RNN and CSV inputs were not reproduced.
- **Distribution shift:** cohort heterogeneity is real (diacetyl point
  estimate +0.062, CI including zero; octanol null) and external transfer is
  inconsistent.  It may explain some variation, but it does not yield a robust
  general signal.
- **Genuine absence:** not proven.  The study is bounded by 56 animals, 945
  events, sparse identified labels and observational data.  The correct claim
  is an absence of evidence for incremental prospective decodability at this
  resolution.

## Research recommendation and 6-ECTS scope

Do not advance to a 6-ECTS causal/whole-brain intervention claim on this
evidence.  The defensible 6-ECTS scope is a methods-and-null-result package
with a focused follow-up:

1. A balanced, animal-disjoint replication with enough dorsal events per
   animal.
2. The paper's exact four-frame alignment and RNN, if the derived CSVs can be
   obtained.
3. Trial-level optogenetic or mutant recordings with sham/off-target controls.
4. Pre-registered event-level metrics and independent cohort replication.

Only after a prospectively validated signal should a 6-ECTS perturbation or
control program be considered.

## Transparency and deviations

- This is not an exact reproduction of Kramer et al.; the published decoding
  CSVs and RNN were unavailable, and no trial-level optogenetic data exist in
  the archive.
- D/V labels are inferred from `dv`; there are no explicit labels.
- Angular velocity comes from the HDF5 behaviour group, and `head_angle` uses
  orientation-corrected `dv`.
- The random-control and permutation counts were reduced for runtime
  (6 label permutations, 24 matched random ablations); they are diagnostics,
  not definitive randomisation tests.
- The diacetyl cohort result is exploratory and would not survive a
  multiplicity correction across all secondary analyses.
- Segment selection is event-aligned rather than online-triggered; every
  feature frame is pre-offset, but the choice of a mid-reversal window uses the
  completed event length.  This is a limitation of the offline benchmark.
- All nulls are preserved.  No result was selected post hoc as primary.
