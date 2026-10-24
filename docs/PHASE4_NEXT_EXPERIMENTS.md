# Phase 4 next experiments

## Priority 0: make the negative result definitive

1. **Exact alignment reproduction.** Obtain the published `SAAV_dorsal.csv`,
   `SAAV_ventral.csv`, `HC_dorsal.csv` and `HC_ventral.csv`, or reconstruct
   their four-frame alignment from the released code and raw archive.  Run the
   published RNN with animal-disjoint folds.
   - Success: reproduce the paper's reported neural gain on its own inputs.
   - Failure: a bounded null under the intended analysis, which is stronger
     than the current independent benchmark.

2. **Balanced animal-disjoint replication.** Pre-register a cohort with enough
   dorsal events per animal (target ≥20 per class per animal) and a behaviour
   baseline that is not near chance.  Report AUROC, AUPRC, log loss, Brier and
   calibration with animal-cluster uncertainty.

## Priority 1: distinguish representation from signal

3. **Matched sequence model.** Use the same four-frame windows with a small
   GRU/TCN and the same animal-disjoint folds, plus a parameter-matched
   behaviour-only model.  This tests representation capacity without changing
   the target.

4. **Neuron-identity permutation and ablation at the sequence level.** Repeat
   the current controls with the sequence model; require the observed effect
   to exceed the random-neuron null before any identity claim.

5. **SAAV-specialised cohort.** Analyse only animals with high-confidence SAAV
   and balanced turn directions; report the selection rule in advance.

## Priority 2: distribution shift and behaviour

6. **Cohort×odour interaction.** Test whether the diacetyl hint replicates in
   an independent diacetyl cohort with matched behaviour variance.  Treat it
   as exploratory until then.

7. **Behaviour decomposition.** Separate head angle, angular velocity and
   body bend contributions at −4 to −2 frames.  Determine whether the residual
   neural signal is a nonlinear function of the same behaviour.

8. **Temporal generalization.** Train at one pre-offset lag and test at
   another; a genuine prospective code should degrade gracefully, not depend
   on one frame.

## Priority 3: causal evidence

9. **Trial-level optogenetics.** Obtain or generate per-trial recordings with
   identified SAAV/RMD/SMD/RIV cells, sham and off-target controls, and
   randomised stimulation.  Pre-register animal-disjoint event-level outcomes.

10. **Mutant controls with matched batches.** If tdc1 is used, match recording
    batch, odour, behaviour and animal size; do not attribute a cohort
    difference to the gene without these controls.

## 6-ECTS scope

Do not start a 6-ECTS causal/whole-brain program on the current evidence.
The defensible 6-ECTS deliverable is the null-result and methods package plus
items 1–3.  Re-evaluate after an exact reproduction or a prospectively
validated, independently replicated signal.
