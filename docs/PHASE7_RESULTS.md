# Phase 7 results — Active Discrepancy Atlas

**Warning:** MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.

## Decision

**H1 NOT SUPPORTED.** The preregistered hybrid policy did not meet the +0.02 absolute gate with a 95% CI above zero at the 40% target budget. No secondary result is promoted to a primary claim.

Hybrid minus random at 40%: **0.0094** (95% block-bootstrap CI -0.0005 to 0.0205; 6/10 paired blocks positive).

The active policies were directionally better than random at 40% in the point estimate, but the preregistered effect size and CI gate were not met. This is a negative primary result, not a licence to promote the best secondary strategy.

## Primary budget (40% of all targets)

| Strategy | Pair Spearman | Profile / ceiling | 90% calibrated coverage | Interval score |
|---|---:|---:|---:|---:|
| uncertainty | 0.0851 | 1.0401 | 0.8999 | 29.66 |
| diversity | 0.0847 | 1.1212 | 0.9000 | 29.65 |
| hybrid | 0.0839 | 1.0796 | 0.8998 | 29.67 |
| effect | 0.0821 | 1.1208 | 0.9000 | 29.68 |
| discrepancy | 0.0763 | 1.0916 | 0.9002 | 29.73 |
| random | 0.0745 | 0.9882 | 0.8998 | 29.70 |

Full-atlas reference (60% budget, 5 seeds): **0.1007** pair Spearman.

Median targets to reach 90% of the full-atlas score: **discrepancy: not reached, diversity: not reached, effect: not reached, hybrid: 117, random: not reached, uncertainty: 130**. `not reached` means the strategy did not reach 90% of the full-atlas score by the 50% budget.

The `Profile / ceiling` column is unstable in this run: several values exceed 1, which means the split-half ceiling (10 draws, `min_events=2`) is underestimated for these test targets. The raw profile Spearman is the interpretable quantity; the normalised value is not used as a success criterion or a claim that the model beats the measurement ceiling.

## Budget curves

| Budget | Strategy | Pair Spearman | Targets |
|---:|---|---:|---:|
| 0.20 | discrepancy | 0.0438 | 52 |
| 0.20 | diversity | 0.0438 | 52 |
| 0.20 | effect | 0.0438 | 52 |
| 0.20 | hybrid | 0.0438 | 52 |
| 0.20 | random | 0.0438 | 52 |
| 0.20 | uncertainty | 0.0438 | 52 |
| 0.30 | discrepancy | 0.0658 | 78 |
| 0.30 | diversity | 0.0734 | 78 |
| 0.30 | effect | 0.0707 | 78 |
| 0.30 | hybrid | 0.0779 | 78 |
| 0.30 | random | 0.0647 | 78 |
| 0.30 | uncertainty | 0.0756 | 78 |
| 0.40 | discrepancy | 0.0763 | 104 |
| 0.40 | diversity | 0.0847 | 104 |
| 0.40 | effect | 0.0821 | 104 |
| 0.40 | hybrid | 0.0839 | 104 |
| 0.40 | random | 0.0745 | 104 |
| 0.40 | uncertainty | 0.0851 | 104 |
| 0.50 | discrepancy | 0.0830 | 130 |
| 0.50 | diversity | 0.0869 | 130 |
| 0.50 | effect | 0.0906 | 130 |
| 0.50 | hybrid | 0.0910 | 130 |
| 0.50 | random | 0.0817 | 130 |
| 0.50 | uncertainty | 0.0903 | 130 |
| 0.60 | full_atlas | 0.1007 | 155 |

At the 20% budget all strategies share the same stratified random seed set by construction; differences begin only after the first acquisition step. The 60% full-atlas reference is a separate five-seed model, not an acquisition strategy.

## Paired deltas versus random

| Budget | Strategy | Delta | 95% CI | Wins / blocks |
|---:|---|---:|---|---:|
| 0.20 | discrepancy | 0.0000 | 0.0000 to 0.0000 | 0/10 |
| 0.20 | diversity | 0.0000 | 0.0000 to 0.0000 | 0/10 |
| 0.20 | effect | 0.0000 | 0.0000 to 0.0000 | 0/10 |
| 0.20 | hybrid | 0.0000 | 0.0000 to 0.0000 | 0/10 |
| 0.20 | uncertainty | 0.0000 | 0.0000 to 0.0000 | 0/10 |
| 0.30 | discrepancy | 0.0011 | -0.0084 to 0.0109 | 4/10 |
| 0.30 | diversity | 0.0087 | -0.0023 to 0.0188 | 6/10 |
| 0.30 | effect | 0.0060 | -0.0048 to 0.0154 | 6/10 |
| 0.30 | hybrid | 0.0132 | 0.0020 to 0.0221 | 9/10 |
| 0.30 | uncertainty | 0.0109 | 0.0024 to 0.0187 | 8/10 |
| 0.40 | discrepancy | 0.0018 | -0.0102 to 0.0148 | 4/10 |
| 0.40 | diversity | 0.0101 | 0.0017 to 0.0195 | 7/10 |
| 0.40 | effect | 0.0076 | -0.0016 to 0.0174 | 6/10 |
| 0.40 | hybrid | 0.0094 | -0.0005 to 0.0205 | 6/10 |
| 0.40 | uncertainty | 0.0106 | 0.0027 to 0.0196 | 7/10 |
| 0.50 | discrepancy | 0.0013 | -0.0096 to 0.0119 | 6/10 |
| 0.50 | diversity | 0.0052 | -0.0010 to 0.0132 | 6/10 |
| 0.50 | effect | 0.0089 | 0.0028 to 0.0158 | 7/10 |
| 0.50 | hybrid | 0.0093 | 0.0016 to 0.0188 | 7/10 |
| 0.50 | uncertainty | 0.0086 | -0.0002 to 0.0187 | 7/10 |

## Anatomical association and anatomy-unexplained readout

| Strategy | Direct-edge AUROC | Anatomy-unexplained AUROC | Pairs |
|---|---:|---:|---:|
| diversity | 0.5800 | 0.6123 | 40341 |
| discrepancy | 0.5733 | 0.6022 | 40341 |
| effect | 0.5706 | 0.6099 | 40341 |
| uncertainty | 0.5663 | 0.6044 | 40341 |
| random | 0.5642 | 0.5926 | 40341 |
| hybrid | 0.5443 | 0.6064 | 40341 |

H2 discrepancy minus uncertainty on anatomy-unexplained AUROC: **-0.0022**; supported: **False**.
H4 hybrid minus random on direct-edge AUROC: **-0.0198**; supported: **False**.

The anatomy-unexplained label means no direct chemical/electrical edge and an observed response in the top decile. It is an association candidate, not evidence of an extrasynaptic mechanism.

## H3 state-conditioned ablation

Mean state minus base pair Spearman: **0.0054** over 3 folds; supported: **False**.

The state ablation is separate from the primary H1 benchmark.

## What this does and does not show

- It is a pseudo-prospective computational benchmark on held-out targets from the same atlas.
- It is not a wet-lab closed loop.
- It does not establish causal effects or a causal graph.
- It does not establish cross-dataset or cross-species transfer.
- The final target list is a set of hypotheses for a future experiment.

## Artifacts

- `reports/phase7/metrics.csv`, `budget_curves.csv`, `deltas.csv`, `targets_to_threshold.csv`
- `reports/phase7/selection_log.jsonl`, `audit.json`, `predictions.parquet`
- `reports/phase7/connectivity/edge_metrics.csv`, `discrepancy_pairs.csv`
- `reports/phase7/design/target_candidates.csv`

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
