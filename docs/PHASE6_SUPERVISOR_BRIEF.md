> **Superseded ceiling (Phase 9).** The split-half reliability ceiling used in
> this document (about 0.152) is biased downward. The variance-corrected
> single-animal ceiling is 0.473 and the repeat ceiling is 0.493. Percentages
> of the ceiling quoted below are therefore too high. See
> [`CORRECTIONS.md`](CORRECTIONS.md) and [`PHASE9_RESULTS.md`](PHASE9_RESULTS.md).
> The historical numbers are retained for audit.

# Phase 6 supervisor brief

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.** The
> recommendation below is a research next step, not an interventional result.

**Question.** Can a compact, feature-conditioned low-rank operator predict the
downstream response to optogenetic stimulation of a neuron it has never been
trained on?

**Data.** Randi et al. (2023), local candidate table: 110 animals, 3,371 events,
237,639 event-level rows, 259 targets, 305 responders, 28,000 pairs with ≥2
events. The local table is a superset of the paper's 23,433 filtered pairs; exact
reproduction is not claimed.

**Design.** Five-fold `unseen_targets` split, five seeds, pair-level Spearman on
`response_mean`, animal-cluster bootstrap, deployable baselines fitted on training
rows only. Preregistered before the full run; deviations logged.

**Primary result.**

| | CRO | additive/responder mean | Δ (95% CI) |
|---|---:|---:|---:|
| Pair Spearman | **0.098** | 0.054 | **+0.044** (+0.028 to +0.059) |

The direction is positive and excludes zero, but the preregistered **+0.05**
effect-size gate is **not met**. Calibration also misses the 0.85–0.95 target
(raw coverage 0.984). Under the frozen rule the strict claim is **NOT SUPPORTED**.

**Why the result is still useful.** Split-half measurement error is severe. The
maximum expected profile correlation is about **0.30**; the CRO reaches **0.094**
(≈64% of that ceiling) while the strongest baseline reaches **0.044** (≈11%).
In other words, the operator recovers most of the *reliably measurable*
target-specific response structure, even though the raw Δ misses an arbitrary
absolute threshold.

**What it does not do.** It does **not** recover direct anatomical connectivity:
CRO AUROC 0.509 versus 0.762 for a connectome ridge. It does **not** establish
causation, and it does **not** transfer across datasets (no external
trial-level cohort is available).

**Recommendation.** Treat Phase 6 as a positive profile-generalisation result and
a negative connectivity/causal-discovery result. The next experiment is a
preregistered replication in new animals with validation-calibrated intervals and
a prospective test of the top model-designed targets. Do not describe the
operator as a causal atlas.
