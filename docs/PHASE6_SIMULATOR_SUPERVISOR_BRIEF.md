# Phase 6 simulator: supervisor brief

## Bottom line

The Phase 6 simulator is **functionally working but scientifically unvalidated**. It produces stable autonomous rollouts, but no learned family beats a constant-state baseline by median skill on the validation block, and no model reproduces the joint distribution or branching statistics better than simple baselines. The honest deliverable is a reproducible negative/limited result with a working perturbation sandbox, not a validated digital twin.

## Strongest evidence

- G1 pass: 5/5 worms, finite H=100 and H=200 trajectories from real histories.
- G2 fail: validation median skill negative for VAR (−0.221), stochastic VAR (−0.782), switching VAR (−3.308).
- G3 fail: 0/5 worms improve energy, autocorrelation and covariance simultaneously.
- G4 fail: branching TV 0.513 vs 0.264 (Markov) and 0.245 (marginal).
- G5 pass: 36/36 perturbation conditions stay inside calibrated support.
- G6/G7 not evaluable: no multi-family specificity test, no matched intervention data.

## Most important correction

Phase 5 selected a residual MLP on one-step validation \(R^2\). In free running it has H20 skill −1.36 and H100 skill −18.0, with behaviour readout supported in only 21.5% of H100 states. This is the clearest demonstration that the old evaluation was not measuring dynamical validity.

## What not to say

- Do not say "the model captures whole-brain dynamics".
- Do not say "stochastic branching is realistic".
- Do not say "AVEL/AVER/RIV perturbation causes ...".
- Do not say "the digital twin predicts stimulation".
- Do not call the test confirmatory: block 2 was reused.

## Decision request

Approve one of:

1. Independent cohort first (recommended): freeze the contract and test VAR vs persistence on new animals.
2. Mechanism follow-up: fit a low-rank switching model only after a new validation cohort is available.
3. Intervention collaboration: obtain matched stimulation/control data before any perturbation claim.

## Deliverables

`docs/SIMULATOR_PHASE6_RESULTS.md`, the nine focused Phase 6 documents, `reports/simulator/phase6/`, `figures/simulator/phase6/`, model cards, and the reworked Core Dynamics Simulator tab.
