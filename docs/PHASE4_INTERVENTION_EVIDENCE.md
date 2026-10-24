# Phase 4 intervention evidence

## Bottom line

**No trial-level intervention effect can be estimated from this archive.**
There are no raw single-cell optogenetic trial recordings in the Dryad
deposit.  The tdc1 cohort is a mutant comparison, not a controlled
perturbation of the recorded circuit.  No virtual perturbation is presented as
a causal result.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

This warning applies to any model-based perturbation display.  It is not a
substitute for an experiment.

## What the paper provides

Kramer et al. report aggregate optogenetic evidence for SMB, SAA, SMD and RIV
neurons using GtACR2, with plate-level sample sizes and population-level
statistics.  Those data are not distributed as trial-level recordings with
per-animal neural activity and behaviour.  Reconstructing trial-level
perturbations from the paper's summary figures would be fabrication.

The published decoding repository contains code that expects pre-extracted
SAAV/HC CSVs.  Those CSVs are not in the repository or the Dryad archive.
The archive's HDF5/MAT files support observational decoding, not an exact
reconstruction of the paper's RNN input.

## What Phase 4 did instead

- Tested observational turn-direction prediction with animal-disjoint splits.
- Treated the tdc1 mutant as a distribution-shift cohort: fit on WT, score
  tdc1, and compare with within-WT transfer.  This is not a causal mutant
  effect because genetic background, recording batch and behaviour differ.
- Ran model-dependent feature ablations.  These ask whether a model uses a
  feature; they do not ask whether an animal needs the neuron.
- Did not run a virtual stimulation experiment.  A model that cannot decode a
  prospective signal has no basis for a credible virtual perturbation.

## tdc1 as distribution shift, not intervention

The tdc1 cohort has 17 animals, 282 long valid events and a dorsal fraction of
29.4%.  WT → tdc1 transfer is modest (behaviour AUROC 0.621; +population
0.614), and octanol → tdc1 is similar (0.612; +population 0.626).  There is no
uncertainty-bounded evidence of a neural gain, and no basis to attribute a
difference to the mutation rather than cohort/batch shift.

## What would be needed for a causal claim

1. Trial-level recordings with identified neurons and behaviour.
2. Randomised stimulation conditions, including sham and off-target controls.
3. Animal-disjoint, event-level analysis with pre-registered outcomes.
4. A positive prospective decoding result before testing model-based
   interventions.
5. Independent replication in a second lab or cohort.

Until then, the defensible conclusion is observational and predictive, not
causal.
