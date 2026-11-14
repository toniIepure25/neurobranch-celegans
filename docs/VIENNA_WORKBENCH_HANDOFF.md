# Handoff memo — Vienna Neuroinformatics Studio

**To:** Denis Kovačević, Research Group Neuroinformatics, University of Vienna
**From:** NeuroBranch engineering
**Date:** 2026-11-11
**Status:** working research product; not a validated digital twin

## What you can inspect now

```bash
python scripts/fetch_vienna_artifacts.py
python -m streamlit run src/neurobranch/bundlenet_sim/app.py
```

Sidebar workspace (each section states its own claim level; the overview includes a ready/missing
delivery snapshot and the product brief is downloadable in the sidebar):

1. **Overview** — product, headline metrics, claim boundaries.
2. **Scope & decisions** — the six Denis decisions, recorded without inferred defaults; exports
   stay marked as unconfirmed until the signed brief is returned, **claim-readiness advisories**
   name any prerequisite the selected scope still lacks, JSON export/import is bound by a
   SHA-256 scope digest so an edited transcription is rejected, and a returned signed block is
   checked for fields, date and digest without claiming to authenticate the signer.
3. **BunDLe-Net** — five released 3-D embeddings; manifold walk, switching linear and quadratic
   models; held-out-worm benchmark with CRPS, 90% coverage, behaviour NLL and horizon curves;
   32-sample predictive spread.
4. **NC-MCM** — choose K, transition matrix, state sequence, behaviour composition, Dirichlet
   transition intervals, model-based intervention trajectories, cross-worm behaviour divergence,
   **assumption audit**.
5. **Premotor** — two modes: **published clamped core** (15 intrinsic integrated, 69 signals
   clamped to recording) and the exploratory 205-node free-run. Clamped mode supports signal
   pulses, intrinsic residual plots and a validation report.
6. **Evidence report** — one-click Markdown report with every number and caveat.
7. **Evidence & lab** — claim ledger, map to the group's published work, next experiments.
8. **Provenance** — citations and machine-readable run provenance.

## What is measured, not assumed

- Tier 1 one-step latent R² = **0.998** leave-one-worm-out; behaviour decode from true latents =
  **96.0%**.
- Open-loop benchmark v3, 25-frame horizon, 16 seeds: manifold walk **83.8%** behaviour accuracy,
  CRPS **0.0969**, NLL **0.542**, latent MSE h10 **0.0445**. Quadratic 82.8% / 0.1150 / 0.618 /
  0.0539; switching 66.8% / 0.1212 / 0.814 / 0.0687.
- **Intervals under-cover:** raw nominal 90% coverage is 68–75%; nested training-worm conformal
  widening raises it to 80–87%, still below nominal. Bands are descriptive, not calibrated.
- NC-MCM audit at K=6: behaviour transport, state–behaviour MI and K sensitivity pass; **strict
  first-order Markov and stationarity fail**. An order-2 temporal-holdout follow-up lowers NLL in
  **5/5** worms (aggregate ΔNLL 0.018, 95% CI 0.005–0.034), but only 2/5 individually and it is
  predictive, not causal. `do(state)` is therefore not an identified effect.
- Premotor canonical recording maps to **84 core / 15 intrinsic / 69 signal**; clamped β=10 run is
  stable (max |u| 3.337). Fresh Julia runs on **all three validation recordings** match the Python
  port at mean trace **r=0.9999998**, max |Δ|=**0.0133**, and behaviour-label parity **99.98%**.
  Behaviour agreement with the recorded readout is **60.2%** canonical, **58.3%** mean across three
  recordings (JS 0.0110). Implementation-faithful replay; still not a digital-twin claim.
- **Independent reversal check:** the paper's neural readout reaches **mean AUC 0.848 / 75.9%**
  binary agreement against the release's `reversal_events`. The clamped model reaches **lag-0 AUC
  0.633 / 55.9%**. Leave-one-recording-out retrospective alignment at **+7.7 frames (~77 ms)**
  raises AUC to **0.834 / 70.6%**, but the mirrored prospective AUC is **0.388** (exploratory
  negative-lag maximum 0.526–0.627, still below persistence), and one-frame persistence reaches
  **0.939 / 94.7%**. The replay carries delayed behavioural information, has no validated forecast,
  and does not beat persistence. This is not a validated behaviour model.
- The external report includes block-bootstrap CIs, event-level recall, β=0 and zero-signal
  controls, and offset/interval sensitivity. Zero signals collapse lag-0 AUC to ~0.50; β=0 is not
  consistently worse, so the behavioural check does not establish a necessary gap-junction role.
- All **913/913** non-zero `A` entries lie inside the transposed chemical-synapse support.
- Correcting the port required four fixes: 1-based Julia label indexing, a one-frame (not
  two-frame) crop per edge, `A`/gap-sign orientation, and the exact 5-forward/6-reverse
  notebook readout (the earlier 4/4 symmetric proxy had the biological sign inverted).
  Tests 40/40 pass locally and on the pod.

## What is still missing

1. **Intervention data.** Without matched stimulation/ablation recordings, NC-MCM `do()` remains a
   Markov counterfactual and the clamped pulse panel remains a simulation. The randomized request
   and analysis plan are ready in `docs/VIENNA_INTERVENTION_PROTOCOL.md`.
2. **A sixth worm / external cohort.** n=5 is enough for a prototype, not a generalization claim.
   The ingestion gate and pre-registration are ready in `docs/VIENNA_NEXT_WORM.md`.
3. **A prioritised scientific question** from you: behaviour switching, individual variability, or a
   specific perturbation.
4. **Fully calibrated intervals.** Conformal widening improves 90% coverage to 80–87%, but does not
   reach nominal and is not worm-conditional.
5. **Independent eight-behaviour labels and BCI/EEG/MeDIL integration.** Not present; do not claim
   them (`docs/VIENNA_SCOPE.md`).

## Suggested first collaboration experiment

Pre-register a held-out-worm comparison on the same behaviour epochs:

- baselines: constant initial state, one-step persistence, majority behaviour;
- model A: manifold walk (descriptive);
- model B: NC-MCM state model at the K selected by the Markov test;
- model C: clamped premotor simulation.

Report per-worm skill, calibration, and behaviour-transition statistics; do not pool worms or
select on the test set. A negative result is informative.

## Boundaries

- BunDLe-Net is a **bioRxiv preprint** (`10.1101/2023.08.08.551978`), not a peer-reviewed article.
- NC-MCM (`10.1371/journal.pcbi.1012674`) is peer-reviewed; its causal claims depend on assumptions
  and are not established by the precomputed artifact.
- The premotor paper (`10.1371/journal.pcbi.1013818`) is peer-reviewed; our clamped mode matches the
  authors' Julia implementation, not an independently validated pointwise replay of the biology.
- No tier establishes whole-brain mechanism, a digital twin, or causality.

## Files to read next

- [`VIENNA_PRODUCT_BRIEF.pdf`](VIENNA_PRODUCT_BRIEF.pdf) — brief for the first meeting.
- [`VIENNA_WORKBENCH.md`](VIENNA_WORKBENCH.md) — architecture, commands, measured results.
- [`VIENNA_MODEL_CARD.md`](VIENNA_MODEL_CARD.md) and [`VIENNA_DATA_CARD.md`](VIENNA_DATA_CARD.md).
- [`LAB_ALIGNMENT.md`](LAB_ALIGNMENT.md) — mapping to the group's work.
- [`VIENNA_WORKBENCH_EVIDENCE.md`](VIENNA_WORKBENCH_EVIDENCE.md) — source-to-claim ledger.
- [`../src/neurobranch/bundlenet_sim/workbench.py`](../src/neurobranch/bundlenet_sim/workbench.py) — Python API.
