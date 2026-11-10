# Lab alignment — Research Group Neuroinformatics, University of Vienna

The product starts from the group's **published** work. It does not attribute BunDLe-Net or NC-MCM
to Denis Kovačević, and it does not invent an internal Denis paper.

## Mapping

| Group output | Product tier | What the product adds | Claim boundary |
|---|---|---|---|
| BunDLe-Net (Kumar, Gilra, Meunier, Grosse-Wentrup; bioRxiv `10.1101/2023.08.08.551978`) | Tier 1 | held-out-worm benchmark, CRPS/coverage/NLL, 32-sample predictive spread, per-worm table | Latent manifold, not mechanism |
| NC-MCM (Grosse-Wentrup, Kumar, Meunier, Zimmer; PLoS Comput Biol `10.1371/journal.pcbi.1012674`) | Tier 2 | assumption audit, transport test, K sensitivity, explicit `do()` caveat | Observational; strict Markov/stationarity currently fail |
| Celegans_premotor (Morrison & Young; PLoS Comput Biol `10.1371/journal.pcbi.1013818`) | Tier 3 | published 84/15/69 clamped mode, signal pulses, validation report, Julia-reference parity, independent reversal-event check | Replay/sandbox; implementation-faithful, not a digital twin |
| Causal-mapping project (2024–2028) | Evidence & lab | pre-registered next experiments, held-out sixth worm, intervention request | No new causal claim |
| BAI/EEGChat, MeDIL | not in v2 | noted as future extensions | Not implemented |

## Group research identity

The group's stated question is how neural activity gives rise to cognition, framed as a causal
question, using calcium imaging, EEG/ECoG, fMRI, and BCI translation. The product is aligned with
that identity by making **assumption audits and claim levels** first-class, not by adding a
generic "digital twin" label.

- Group site: https://ni.cs.univie.ac.at/
- Projects: https://ni.cs.univie.ac.at/research/projects
- Code: https://ni.cs.univie.ac.at/research/code
- Denis Kovačević team page: https://ni.cs.univie.ac.at/about-us/team/person/unitid/81/denis-kovacevic

## What to ask Denis in the first meeting

1. Which of the three objects is closest to the lab's next grant or paper?
2. Is a sixth held-out worm available, and what is its recording/protocol?
3. Is there an intervention dataset (optogenetic, ablation, microfluidic) that could test a
   pre-registered `do()` claim?
4. Should the next iteration target a higher-order NC-MCM model, a better calibrated Tier-1
   generator, or a perturbation experiment that can test the now-matched clamped replay? The
   Julia-reference parity check is already complete for all three validation recordings.
