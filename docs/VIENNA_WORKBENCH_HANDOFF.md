# Handoff memo — Vienna Neuroinformatics Workbench

**To:** Denis Kovačević, Research Group Neuroinformatics, University of Vienna
**From:** NeuroBranch engineering
**Date:** 2026-11-09
**Status:** research prototype; not a validated digital twin

## What you can inspect now

```bash
python scripts/fetch_vienna_artifacts.py
python -m streamlit run src/neurobranch/bundlenet_sim/app.py
```

Open the printed local URL. Four tabs:

1. **BunDLe-Net** — five released 3-D embeddings, held-out-worm metrics, stable
   observed-transition walk, unstable quadratic forecast for comparison.
2. **NC-MCM** — choose K, inspect cognitive-state transitions, Markov p-values,
   behaviour composition, and an explicitly observational state read-out.
3. **Premotor** — 205-node fitted chemical matrix integrated with the published
   cubic rate ODE; stimulate/ablate nodes.
4. **Provenance** — citations, claim levels, missing pieces.

## What is measured, not assumed

- Tier 1 one-step latent R² = **0.998** leave-one-worm-out.
- Tier 1 behaviour decode from true latents = **96.0%**.
- Tier 1 behaviour decode from simulated quadratic latents = **33.5%** vs **29.8%**
  majority; one of five free-run folds has negative skill.
- Therefore the default simulator is a **manifold walk over observed transitions**,
  not the quadratic model. It is stable by construction and does not extrapolate.

## What we need from the group

1. **Premotor export:** the gap-junction matrix `W` and the neuron-name order used
   for `A_10takes_avg`. Without them Tier 3 is chemical-only and anonymous.
2. **Scientific target:** which question should the workbench answer first —
   behaviour switching, individual variability, or a specific perturbation?
3. **Intervention data:** any matched stimulation/ablation recordings, so the
   NC-MCM `do()` panel can become an actual effect estimate rather than a
   conditional read-out.
4. **Sixth worm / external cohort:** to test whether the Tier 1 manifold walk
   generalizes beyond the five released embeddings.

## Suggested first collaboration experiment

Pre-register a held-out-worm comparison on the same behaviour epochs:

- baseline: constant initial state and one-step persistence;
- model A: manifold walk (descriptive);
- model B: NC-MCM state model at the K selected by the Markov test;
- model C: premotor ODE with the full `W` matrix.

Report per-worm skill, calibration, and behaviour-transition statistics; do not
pool worms or select on the test set. A negative result is informative and should
be reported as such.

## Boundaries

- BunDLe-Net is a **bioRxiv preprint** (`10.1101/2023.08.08.551978`), not a
  peer-reviewed journal article.
- NC-MCM (`10.1371/journal.pcbi.1012674`) is peer-reviewed; its causal claims
  depend on assumptions and are not established by the precomputed artifact.
- The premotor paper (`10.1371/journal.pcbi.1013818`) is peer-reviewed; our Tier 3
  is a reduced, incomplete export of it.
- No tier establishes whole-brain mechanism, a digital twin, or causality.

## Files to read next

- [`VIENNA_WORKBENCH.md`](VIENNA_WORKBENCH.md) — architecture, commands, metrics.
- [`VIENNA_WORKBENCH_EVIDENCE.md`](VIENNA_WORKBENCH_EVIDENCE.md) — source-to-claim ledger.
- [`../src/neurobranch/bundlenet_sim/workbench.py`](../src/neurobranch/bundlenet_sim/workbench.py) — Python API.
