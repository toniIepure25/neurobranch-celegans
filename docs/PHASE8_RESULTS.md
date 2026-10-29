# Phase 8 results — Adaptive Perturbation Twin

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Run

| Item | Value |
|---|---|
| Split | `unseen_animals`, seed **41** |
| Folds | 5 (61–63 train, 20–21 validation, 20–21 test animals) |
| Events | 3,371 total; animal-held-out test tasks use target-disjoint support/test sets |
| Model seeds | 17, 23, 31 (deep ensemble) |
| Acquisition seeds | 101, 202 |
| Support sizes | 0, 2, 4, 8, 16; primary k=8 |
| Row-level predictions | 930,846 |
| Bootstrap | 100 animal-cluster replicates in the recovered summary (preregistered 1,000; see deviations) |
| Leakage audit | **clean** — no animal, support/test target, support/test event, or test-event overlap |
| Calibration | validation-only scale 0.2626 |

The full predictor export, metrics, selection log, models and cache are on disk
but row-level/raw-derived artifacts are deliberately git-ignored under the Randi
licence. The committed artifacts are aggregate: `summary.json`, `readout.json`,
`metrics.csv`, `audit.json`, `README.md`, `manifest.json`.

## Hypothesis decisions

| Hypothesis | Result | Decision |
|---|---|---|
| **H1** APT(k=0) vs static CRO, pair Spearman | **+0.0442**, 95% CI **[+0.0080, +0.0901]**, 5/5 fold wins | **pair gate passed** |
| H1 trajectory vs per-responder template | **−0.0000066** (twin 0.03350, template 0.03350) | **trajectory gate failed** |
| **H1 overall** | pair passed, trajectory failed | **not supported** |
| **H2** APT(k=8 random) vs APT(k=0) | **+0.0022**, CI **[−0.0046, +0.0118]**, 3/5 folds | **not supported** |
| **H3** EIG proxy vs random at k=8 | **−0.0038**, CI **[−0.0070, +0.0026]**, 3/5 folds | **not supported** (and H2 was not supported, so H3 is descriptive only) |
| **H4** 3-event sequence rollout vs persistence | trajectory **+0.0230**; coverage **0.9037** | **not supported** (gate requires +0.050; coverage gate passed) |

### What H1 actually shows

The primary scalar endpoint is a clear, fold-consistent improvement over the
static Phase 6 CRO: the animal-held-out, target-disjoint APT beats CRO on pair
Spearman in all five folds. That is a real model-dependent predictive gain.

It is **not** a trajectory gain. The residual trajectory head converged to the
train-only per-responder template, so the trajectory correlation is numerically
identical to the template baseline. The preregistered H1 required both pair and
trajectory gates, so the hypothesis is reported as not supported rather than
silently promoting the scalar component.

### What H2/H3 actually show

A support set of 8 events gives a small positive point estimate over no support,
but the animal-cluster interval crosses zero. This is consistent with the
training-only variance decomposition: animal η² ≈ **0.00048**, so the available
animal-level calibration signal is small in this atlas. The result is a bounded
negative about this support design, not evidence that personalisation is
impossible in principle.

The ensemble-disagreement `eig` proxy did not beat random selection. This does
not establish that information-directed acquisition is useless; it establishes
that this proxy and this atlas did not produce a detectable advantage at k=8.

### What H4 actually shows

The sequential rollout beats persistence by +0.023 trajectory correlation with
0.904 coverage, but misses the preregistered +0.050 gate. It is directionally
positive and far from a closed-loop result; previous test outcomes were observed
by design.

## Strategy readout at k=8

| Strategy | Mean pair Spearman | SD | n |
|---|---:|---:|---:|
| diversity | 0.10309 | 0.05926 | 186 |
| hybrid | 0.10299 | 0.05941 | 186 |
| eig (disagreement proxy) | 0.10295 | 0.06007 | 186 |
| uncertainty | 0.10276 | 0.05896 | 186 |
| random | 0.10221 | 0.05957 | 186 |

No active strategy produced a practically meaningful separation from random.
The random budget curve continued to rise at k=16 (0.10533 vs 0.10221 at k=8),
so the support budget, not the selection heuristic, is the stronger lever in
this dataset.

## Honest interpretation

- APT is a **better scalar response operator** than the static CRO under this
  animal-held-out, target-disjoint protocol.
- APT is **not** demonstrated to personalise trajectories or to benefit from
  active support selection in this atlas.
- The sequence readout is directionally positive but below the frozen gate.
- None of this is a causal effect, a wet-lab closed loop, a cross-dataset
  transfer result, or a validated digital twin.

## Deviations that matter for interpretation

- The final split seed is **41**, not the originally planned 37. An initial
  seed-37 run was aborted because active selection was pathologically slow; the
  reported run uses the untouched seed 41. See `PHASE8_DEVIATIONS.md`.
- The summary bootstrap uses **100** animal-cluster replicates after a
  performance recovery; the point estimates and qualitative decisions are
  unchanged, but the intervals are slightly less precise than the preregistered
  1,000.
- H5 (history-feature ablation) was not run and is reported as not run.

## Reproduction

```bash
python scripts/phase8_design.py
python scripts/phase8_run.py --config configs/phase8/apt.yaml
python scripts/phase8_summary.py --n-boot 100
python scripts/phase8_readout.py --output reports/phase8
python scripts/phase8_manifest.py
```

The exact frozen gates and leakage controls are in
`docs/PHASE8_PREREGISTRATION.md`; the evaluation card is
`docs/PHASE8_EVALUATION_CARD.md`.
