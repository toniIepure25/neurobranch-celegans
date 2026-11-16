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
- Denis Kovačević stable team page: https://ni.cs.univie.ac.at/about-us/team/
- Denis Kovačević UCris profile: https://ucrisportal.univie.ac.at/en/persons/denis-kovacevic/
- Public-source audit: [`DENIS_PUBLIC_SOURCE_AUDIT.md`](DENIS_PUBLIC_SOURCE_AUDIT.md)

No publication is listed on the checked group or UCris pages. That is a statement about the pages
checked on 2026-11-13, not proof that no publication exists elsewhere.

## What to ask Denis in the first meeting

The canonical surface is the **Scope & decisions** page. Record the six decisions in writing; do not
infer them from the group page or from this product. The readiness rail, option guidance and live
contract show what each answer commits the project to; they never fill an answer on Denis's behalf.

1. Primary object: causal mechanism, external generalization, reusable workbench, or BCI/NeuroAI?
2. Primary endpoint: released eight states, binary reversal, turn direction, continuous kinematics,
   or another endpoint?
3. Which data actually exist, and under what access conditions?
4. What would count as success: positive result, rigorous negative, reusable benchmark, wet-lab
   design, or working demo?
5. If an intervention is in scope, which circuit, modality and positive control?
6. What is the stop rule, budget and deadline?

A recorded answer is a transcription, not a signature. The signed block in
[`DENIS_REQUIREMENTS_BRIEF.md`](DENIS_REQUIREMENTS_BRIEF.md) remains authoritative.

The room checks consistency as well as completeness. A causal object without a specified target
and a BCI/NeuroAI object with a worm endpoint are shown as blockers. After an export, any changed
decision is listed field by field and blocks the digest stage until a fresh transcription is
downloaded; the provenance fields are not bound by that digest.

The option atlas is an explanation surface, not a seventh answer: it never writes a value or
changes the digest. The downloadable decision packet is a transfer convenience with a hash
manifest; it keeps `unconfirmed_transcription` and is neither evidence nor a signature. Its
source documents and schemas remain in the repository, where they can be reviewed directly.

When the block is returned, the Scope & decisions page checks it in this order: required fields,
`YYYY-MM-DD` date, digest syntax, and equality between the signed digest and the current
transcription digest. A match is reported as **signature fields present — identity not verified**.
A human must then compare `Confirmed by` with the document received from Denis; the upload never
changes the global unconfirmed status or closes G0 by itself.
