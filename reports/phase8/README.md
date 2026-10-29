# Phase 8 — Adaptive Perturbation Twin: readout

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

This is a pseudo-prospective, model-based benchmark. It does not establish
causal effects, a closed-loop experimental result, or a validated digital twin.

## Hypothesis decisions

| Hypothesis | Supported | Effect | Gate |
|---|---|---|---|
| H1 APT(k=0) vs CRO | False | pair Δ 0.044157912132003804, trajectory Δ -6.5921302223048706e-06 | pair Δ ≥ +0.010 & CI low > 0 & trajectory Δ ≥ +0.050 |
| H2 APT(k=8) vs k=0 | False | Δ 0.0022059447757057093 | Δ ≥ +0.010 & CI low > 0 |
| H3 EIG vs random | False | Δ -0.0038279515709690737 | Δ ≥ +0.010 & CI low > 0; confirmatory only if H2 supported |
| H4 sequence rollout | False | trajectory Δ 0.02296379841957328, coverage 0.9036719926074929 | trajectory Δ ≥ +0.050 and coverage 0.80–0.95 |

Leakage audit clean: **True**.

## Strategy readout at k=8

| strategy | mean pair Spearman | std | n |
|---|---|---|---|
| diversity | 0.103091 | 0.059262 | 186 |
| hybrid | 0.102985 | 0.059413 | 186 |
| eig | 0.102949 | 0.060074 | 186 |
| uncertainty | 0.102756 | 0.058963 | 186 |
| random | 0.102206 | 0.059567 | 186 |

## What this does and does not show

- It tests animal-conditional, support-set personalisation under
  target-disjoint support/test splits.
- It does **not** test a wet-lab closed loop, causal graph, cross-dataset
  transfer, or a foundation model.
- `eig` is a deep-ensemble disagreement proxy, not a formal posterior
  expected-information-gain guarantee.
- A null result is reported with intervals, not converted into a claim of
  absence.

## Reproduction

```bash
python scripts/phase8_run.py --config configs/phase8/apt.yaml
python scripts/phase8_readout.py
```
