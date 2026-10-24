# Phase 5 supervisor brief

**To:** Denis Kovačević  
**Project:** NeuroBranch / *C. elegans* turn-direction decoding  
**Branch:** `research/phase5-saav-mechanistic-validation`  
**Status:** reconstruction complete; exact reference reproduction blocked by
missing author inputs; no validated causal model.

## One-paragraph summary

Kramer et al. report positive SAAV + head-curvature decoding of post-reversal
turn direction. Phase 4 found no incremental neural evidence. I inspected the
paper, the official `flavell-lab/SAAV_Decoding` code and the Dryad v5 archive,
reconstructed the method from raw HDF5 data, and ran four offline regimes plus a
genuinely causal-onset online benchmark. **No reconstruction shows a positive,
properly scored, animal-disjoint SAAV increment.** Exact reproduction is
impossible with public artifacts: four aligned CSVs and the preprocessing script
are missing. The leading explanation is an evaluation-unit/preprocessing
difference (section/timestep class-balanced accuracy vs event-level
animal-disjoint scoring), not that either group is wrong. The result is
inconclusive about a modest true effect because the offline cohort has only 62
dorsal events.

## What was done

- Source contract reconstructed from code, including the section-level metric,
  zero hidden state, dropout 0.5, LayerNorm and per-timestep sigmoid.
- 80/82 Dryad files verified by SHA-256; the two missing files are unused.
- 250 phase-aligned wild-type events (62 dorsal/188 ventral, 26 animals) used
  for offline comparison; 428 SAAV-cohort events (27 animals) for online.
- Offline regimes: original-style, animal-disjoint, chronological within-animal,
  external cohort.
- Online landmarks fixed at 0.6–7.2 s after a causal onset; no future frame,
  duration or final direction used to choose the window.
- Mechanism analyses for SAAV, SAAD, SMDV, SMDD, RMDV, RMDD, RIV, SMBV, SMBD,
  with cross-animal stability and temporal controls.
- Intervention evidence map; no pseudo-trials and no virtual intervention.
- Dashboard tab, tests, Ruff, documentation and reproducible reports added.

## Headline numbers

| analysis | result |
|---|---|
| Offline original-style SAAV AUROC | 0.540 (behaviour 0.534); Δ log loss −0.0006 [−0.0029, +0.0018] |
| Offline animal-disjoint | no SAAV increment; grouped AUROC 0.37–0.40 |
| Offline external cohort | adverse transfer (AUROC 0.30–0.33 vs 0.56 behaviour) |
| Online SAAV at 0.6–7.2 s | no log-loss increment; adverse at 4.8 s |
| Online sensory context | AUROC +0.135 to +0.139 over behaviour at 0.6/1.2 s; log-loss CI still crosses zero |
| Online population (label/motor neurons excluded) | AUROC up to +0.12 at 0.6/1.2 s and +0.08 at 7.2 s; **no log-loss increment** |
| Mechanism SAAV | Δ log loss +0.010 [−0.008, +0.029]; no stable cross-animal effect |
| Mechanism SMDV | large exploratory effect, 61 events/6 animals, unstable; not causal |
| Intervention | no trial-level stimulation data; no causal claim |

## Why the discrepancy is not resolved

1. The published metric is **section/timestep class-balanced accuracy** over all
   47 overlapping four-frame sections per event, not event-level animal-disjoint
   AUROC/log loss. Overlapping sections can make a weak event-level signal look
   stronger.
2. The authors' aligned CSVs contain their event extraction, SAAV trace,
   alignment and normalization. None is public. Exact reproduction is
   `BLOCKED_MISSING_INPUTS`.
3. Our reconstruction uses different phase sampling, event selection and a
   PyTorch rather than JAX/Equinox implementation.
4. The SAAV subset (27 animals) behaves differently from the full wild-type set
   (56 animals), so cohort composition matters.
5. Power is limited (62 dorsal offline; approximate power 0.65 at AUROC 0.60).

## What is and is not supported

**Supported observationally:** sensory context and, weakly, population activity
carry direction information early after onset. SAAV does not add proper-score
information in this reconstruction.

**Not supported:** that SAAV causes turning; that the compact circuit is
necessary; that SMDV is a general mechanism; that the published result is wrong;
that any virtual perturbation is causal.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Discussion points for Denis

1. **Metric choice.** Is the field's section-level accuracy the right target, or
   should we insist on event-level animal-disjoint proper scores? If the latter,
   what is the minimum effect size worth pursuing?
2. **Data request.** Do we send the prepared author request now, and who should
   sign it? The four CSVs are the single highest-value next step.
3. **Cohort definition.** Should the primary cohort be all wild-type events
   (more power) or SAAV-identified events (mechanistically cleaner but 27
   animals)? The two give different behaviour baselines.
4. **SMDV.** Is a six-animal exploratory effect worth a targeted follow-up, or
   does it risk a garden-of-forking-paths problem? I would treat it as a
   hypothesis only.
5. **Intervention collaboration.** Is there a realistic path to trial-level
   SAAV stimulation, or should the project stay observational?
6. **6 ECTS scope.** Is the package in `PHASE5_NEXT_EXPERIMENTS.md` realistic,
   and should WP0 (data request/exact rerun) be the only committed deliverable
   until the inputs arrive?

## Recommendation

Do **not** build a virtual intervention or a causal claim on this decoder. Send
the data request, reproduce the official section-level result exactly, and
pre-register the event-level animal-clustered re-analysis on the same arrays.
That is the cleanest test of whether the published signal survives the
evaluation unit change.

## Where to look

- [`PHASE5_RESULTS.md`](PHASE5_RESULTS.md) — full results.
- [`PHASE5_PROTOCOL_COMPARISON.md`](PHASE5_PROTOCOL_COMPARISON.md) — exact
  protocol differences.
- [`PHASE5_ONLINE_DECODING.md`](PHASE5_ONLINE_DECODING.md) — causal-onset test.
- [`PHASE5_NEURAL_MECHANISMS.md`](PHASE5_NEURAL_MECHANISMS.md) — cell analyses.
- [`PHASE5_INTERVENTION_EVIDENCE.md`](PHASE5_INTERVENTION_EVIDENCE.md) — what
  interventions do and do not show.
- [`PHASE5_MISSING_DATA_REQUEST.md`](PHASE5_MISSING_DATA_REQUEST.md) — unsent
  request.
- [`PHASE5_NEXT_EXPERIMENTS.md`](PHASE5_NEXT_EXPERIMENTS.md) — 6-ECTS plan.
