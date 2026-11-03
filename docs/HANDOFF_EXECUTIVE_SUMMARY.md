# NeuroBranch handoff: executive summary

> **Historical J5/J1 handoff.** The current audited result is the B3-vs-B5
> Scientific Studio: paired H20 energy **+0.2644**, conditional interval
> `[+0.0750, +0.4031]`, recording interval `[−0.0408, +0.4793]`, exploratory.
> See [`STUDIO_SCIENTIFIC_EVIDENCE.md`](STUDIO_SCIENTIFIC_EVIDENCE.md). The J5
> coverage values below (including 0.947) belong to that earlier model family
> and must not be presented as the current B3/B5 calibration.

**Purpose.** This is an internal scientific handoff for an MSc discussion with
Denis Kovačević (Neuroinformatics Research Group, University of Vienna). It is
not a thesis and not a systematic review. It is also not a confirmed statement
of his requirements. The original conversation was reconstructed provisionally;
the exact assignment still needs to be agreed.

**Branch.** `research/supervisor-handoff`, created from
`10694a80568c06e6180f6af08bd1a0f2c28e4237` on
`research/joint-dynamics-calibration`. All historical branches, negative
findings, reports and corrections are preserved. No history was rewritten.

## Bottom line

1. The project built a substantial, reproducible research prototype for
   low-dimensional *C. elegans* population dynamics, free-running simulation and
   model-dependent virtual perturbation.
2. It did **not** validate the simulator, recover neuron-specific causal effects,
   reproduce BunDLe-Net, or establish animal-independent generalization.
3. The strongest defensible positive result is narrow: in a fixed 10-D PCA
   fitted on training frames only, a corrected population-mean VAR predicts
   held-out joint neural trajectories better than an independent per-neuron AR
   model at H20. The paired H20 energy improvement is **+0.446
   [0.183, 0.660]** and the trajectory improvement is **+0.375
   [0.218, 0.522]**.
4. The gain is a **conditional-mean** gain. Full covariance hurts, marginal
   per-neuron CRPS is unresolved, and probabilistic calibration fails. The
   model is not a validated generative digital twin.
5. Because the test split has been reused and the five recording sessions are
   not five independent animals, even the central result is **exploratory**.
   It is evidence for a follow-up experiment, not confirmation.

## The central result in one table

Lower energy is better. "Improvement" is reference minus model, so a positive
value favours the population model.

| Quantity | Estimate | 95% block-bootstrap interval | Status |
| --- | ---: | --- | --- |
| J5 vs J1, H20 energy | +0.446 | [0.183, 0.660] | exploratory support |
| J5 vs J1, H20 trajectory | +0.375 | [0.218, 0.522] | exploratory support |
| J5 vs J0 persistence, H20 energy | +0.250 | [0.020, 0.427] | fragile; block-10 interval crosses zero |
| J5 vs J1, full-neuron CRPS | +0.0048 | [-0.0073, 0.0156] | unresolved |
| J5 vs J1, all-neuron energy | +0.203 | [-0.010, 0.442] | unresolved |
| J5 50/80/95 coverage (full neuron) | 0.588 / 0.849 / 0.947 | nominal 0.50 / 0.80 / 0.95 | calibration fails |
| J5 neural covariance error | 0.604 | gate <= 0.50 | fails |

J2 (population mean, parametric diagonal innovations) is effectively identical
to J5: energy 2.700, trajectory 2.058, coverage 0.586 / 0.846 / 0.944. The mean
advantage is therefore not an artefact of one innovation sampler. Both J2 and J5
still fail calibration and covariance fidelity. The marginal CRPS gain is
**unreliable**: its interval includes zero, and it should not be presented as a
per-neuron improvement.

## Evidence status

**Supported within the available design (descriptive or technical):**

- Historical overdispersion was reproduced and its dominant cause was the
  multi-lead residual pool: H20 energy 4.609, 95% coverage 0.999.
- Correcting the pool is reproducible to machine precision (maximum relative
  difference 5.5e-9).
- The one-step population residuals are close to white (median absolute ACF1
  0.022; ACF5 0.009), so a temporal innovation model is not justified.
- Earlier deterministic adaptive VAR improved H20 MSE over the frozen VAR, but
  its stochastic extension did not calibrate.

**Exploratory:**

- All test-split results, including the central contrast, because the test block
  was reused across phases.
- The population conditional mean adds joint information beyond independent AR
  in the fixed 10-D PCA, with consistent direction in 4/5 sessions.
- Session-level consistency: session 1 is negative and session 2 is ambiguous.
- The external Kato Stim pilot: 0/7 sessions beat persistence (median skill
  -0.602), but animal independence is undocumented and the raw file is not in
  this checkout.

**Invalidated, withdrawn or not supported:**

- The 0.560 measurement "ceiling" as a bound. `R(22)` is extrapolated for 87 of
  126 targets; Phase 15 reports `R_obs = 0.2606`, `R_22 = 0.3006`, and
  `R_id = 0.3501` as different estimands.
- Claims that observed trajectories branch more than null models. The evidence
  is an entropy deficit, not excess branching.
- BunDLe-Net as a causally trained, generalizing representation. The distributed
  embedding is transductive and the causal reimplementation did not reproduce
  its geometry.
- Neuron-specific perturbation effects. Corrected random-neuron controls make
  the candidate effect non-specific.
- "Validated digital twin", "causal control", "reliable neuron-specific
  stimulation prediction", and "animal-independent replication".

**Not evaluable:**

- Biological validation of any virtual perturbation: `NOT_EVALUABLE`.
- Current-checkout cross-stimulation refit: `WT_Stim.mat` is absent. Historical
  external-validation work used it from OSF; that pilot is documented but cannot
  be rerun without re-downloading the raw file.

## What to do next

Do not run another architecture search on the reused test split. Choose one of
two bounded 6-ECTS directions:

- **A. Replicate the population-mean result** in a genuinely independent cohort,
  with a prespecified fixed representation, H20 joint energy as the primary
  endpoint, and independent AR plus persistence as baselines.
- **B. Turn intervention-aware simulation into a design problem**: state the
  causal assumptions, audit matched stimulation data, and run a minimal
  falsification test with held-out animals and targets. Do not promise a causal
  simulator within 150 hours.

The immediate discussion questions are in
[HANDOFF_QUESTIONS_FOR_DENIS.md](HANDOFF_QUESTIONS_FOR_DENIS.md). The full
argument and caveats are in [HANDOFF_RESEARCH_MEMO.md](HANDOFF_RESEARCH_MEMO.md).

## Standing labels

Every perturbation output must retain:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation: `NOT_EVALUABLE`.

Final probabilistic label: `PROBABILISTIC CALIBRATION NOT ESTABLISHED`.
