# Handoff memo — Vienna Neuroinformatics Studio

**To:** Denis Kovačević, Research Group Neuroinformatics, University of Vienna
**From:** NeuroBranch engineering
**Date:** 2026-11-09
**Status:** working research product; not a validated digital twin

## What you can inspect now

```bash
python scripts/fetch_vienna_artifacts.py
python -m streamlit run src/neurobranch/bundlenet_sim/app.py
```

Six tabs:

1. **Overview** — what the product is, headline metrics, claim boundaries.
2. **BunDLe-Net** — five released 3-D embeddings; manifold walk, switching linear and quadratic
   models; held-out-worm metrics; three-model open-loop benchmark.
3. **NC-MCM** — choose K, transition matrix, state sequence, behaviour composition, Dirichlet
   transition intervals, model-based intervention trajectories, cross-worm behaviour divergence.
4. **Premotor** — 205-node fitted matrix, **recovered neuron names and W**, named stimulation and
   ablation, activity export, stability warning.
5. **Evidence report** — one-click Markdown report with every number and caveat.
6. **Provenance** — citations and machine-readable run provenance.

## What is measured, not assumed

- Tier 1 one-step latent R² = **0.998** leave-one-worm-out.
- Tier 1 behaviour decode from true latents = **96.0%**.
- Behaviour decode from simulated quadratic latents = **33.5%** vs **29.8%** majority.
- Open-loop benchmark, 25-frame horizon, 4 seeds: manifold walk **70.8%** behaviour accuracy,
  quadratic **64.9%**, switching **50.3%**. Manifold walk also has the lowest latent MSE at h10
  (**0.0517**). It wins because it samples observed transitions; it cannot extrapolate.
- Pod rerun of the benchmark agrees with local means to `< 2e-16`.
- Premotor names/W recovery passes a hard check: all **913/913** non-zero `A` entries lie inside
  the chemical-synapse support.

## What is still missing

1. **Clamped-signal premotor mode.** The paper simulates ~15 intrinsic neurons with 69 signal
   neurons clamped to recordings. The current page runs all 205 nodes freely. At the published
   β=10 that free-run becomes unstable, so the UI defaults to β=0 and warns. This is the
   highest-value next engineering step.
2. **Intervention data.** Without matched stimulation/ablation recordings, NC-MCM `do()` remains a
   Markov counterfactual, not an identified effect.
3. **A sixth worm / external cohort.** n=5 is enough for a prototype, not for a generalization claim.
4. **A prioritised scientific question** from you: behaviour switching, individual variability, or a
   specific perturbation.

## Suggested first collaboration experiment

Pre-register a held-out-worm comparison on the same behaviour epochs:

- baselines: constant initial state, one-step persistence, majority behaviour;
- model A: manifold walk (descriptive);
- model B: NC-MCM state model at the K selected by the Markov test;
- model C: premotor clamped-signal simulation, once implemented.

Report per-worm skill, calibration, and behaviour-transition statistics; do not pool worms or
select on the test set. A negative result is informative.

## Boundaries

- BunDLe-Net is a **bioRxiv preprint** (`10.1101/2023.08.08.551978`), not a peer-reviewed article.
- NC-MCM (`10.1371/journal.pcbi.1012674`) is peer-reviewed; its causal claims depend on assumptions
  and are not established by the precomputed artifact.
- The premotor paper (`10.1371/journal.pcbi.1013818`) is peer-reviewed; our Tier 3 is a free-run
  sandbox, not the paper's clamped-signal model.
- No tier establishes whole-brain mechanism, a digital twin, or causality.

## Files to read next

- [`VIENNA_WORKBENCH.md`](VIENNA_WORKBENCH.md) — architecture, commands, measured results.
- [`VIENNA_BENCHMARK_DESIGN.md`](VIENNA_BENCHMARK_DESIGN.md) — pre-run benchmark protocol.
- [`VIENNA_WORKBENCH_EVIDENCE.md`](VIENNA_WORKBENCH_EVIDENCE.md) — source-to-claim ledger.
- [`../src/neurobranch/bundlenet_sim/workbench.py`](../src/neurobranch/bundlenet_sim/workbench.py) — Python API.
