# Denis requirements brief — draft for confirmation

**Status:** draft, not confirmed. This is the **canonical capture form** for the six decisions,
not a reconstruction of Denis Kovačević's requirements. No answer below may be inferred or filled
in by the project team. The public record is audited separately in
[`DENIS_PUBLIC_SOURCE_AUDIT.md`](DENIS_PUBLIC_SOURCE_AUDIT.md).

**Why this exists.** The repository currently treats the original brief as provisional. The group
page describes an active 2024–2028 project, *An interdisciplinary approach to learn and test the
causal mapping between neural network dynamics and behavior*, but it does not publish a project
brief or a Denis-specific specification. The studio therefore cannot honestly claim to do "exactly
what Denis wants" until the choices below are confirmed. The **Scope & decisions** page records a
transcription; it is not a signature.

## 1. Six decisions that fix the scope

Please answer in writing; one line per question is enough.

1. **Primary object.** Which is primary for the next project?
   - (a) a causal test of a named-neuron or circuit perturbation;
   - (b) external generalization of the representation/forecast;
   - (c) a reusable lab workbench (data intake, QC, annotation, provenance);
   - (d) a BCI/NeuroAI translation project, separate from the worm work.

2. **Primary endpoint.** Which endpoint should be pre-registered as primary?
   - the released eight behavioural states;
   - binary reversal onset/offset;
   - dorsal versus ventral turn;
   - a continuous kinematic variable;
   - another endpoint (please specify).

3. **Data actually available.** Which of these exist, and under what access conditions?
   - a sixth held-out worm with the exact BunDLe-Net consistent embedding;
   - raw traces plus behaviour video for the three validation recordings;
   - a randomized optogenetic/ablation intervention with matched sham;
   - an independent eight-behaviour video annotation;
   - none of the above yet.

4. **Definition of success.** At the end of the project, what would count as success?
   - a positive predictive or causal result;
   - a rigorous negative result;
   - a reusable benchmark/protocol;
   - a design for a wet-lab experiment;
   - a working product demo for the group.

5. **Biological target and intervention.** If an intervention is in scope, which circuit and which
   modality are biologically justified, and which positive control should prove the assay works?
   The current model ranking is not sufficient to choose a target: random control neurons produced
   effects of similar or larger magnitude.

6. **Stop rule and budget.** What is the time/ECTS budget, and what result would make you stop or
   change direction? The existing evidence is mostly negative; a pre-agreed stop rule prevents
   another architecture search on reused data.

## 2. Decision matrix

| If the answer is… | First workstream | Deferred |
|---|---|---|
| causal mechanism | randomized intervention + power + independent labels | BCI; sixth worm as a goal in itself |
| external generalization | sixth worm + independent annotations | intervention; BCI |
| lab workbench | intake, QC, annotation UI, provenance | no new scientific claim |
| BCI/NeuroAI | separate EEG/BCI lane, ethics, BIDS | any mixing with worm claims |
| digital twin | L0–L3 readiness plan; no promise of L3 | twin claim until L2 + L3 |

## 3. Digital-twin readiness levels

The project will use these labels and no stronger one:

| Level | Meaning | Current |
|---|---|---|
| L0 descriptive replay | reproduces published dynamics; no individual calibration | **current** |
| L1 external validity | held-out animal + independent behaviour + calibrated uncertainty | not yet |
| L2 causal validity | randomized intervention, replicated, prospective effect | not yet |
| L3 individual/closed-loop twin | per-animal parameters, online prediction/control | not yet |

A "digital twin" requires **L2 + L3**. The current studio is L0.

## 4. What the project will not do without an answer

- It will not choose a target neuron on the basis of a non-specific model ranking.
- It will not call a model-based counterfactual a causal effect.
- It will not treat released labels as independent video annotations.
- It will not duplicate or resample a worm and call it a sixth worm.
- It will not integrate BCI/EEG/fMRI/MeDIL into the worm claim.

## 5. Confirmation block

```text
Primary object:
Primary endpoint:
Available data:
Success criterion:
Target / intervention (if any):
Stop rule:
Budget / deadline:
Scope digest: ________________________________________________
Confirmed by: ____________________   Date: __________
```

**Return to:** the project owner. The project team will freeze the answers in
`docs/GAP_CLOSURE_MATRIX.md` before collecting or analysing new data.

**A transcription is not a signature.** A team member may record a reported answer on the Scope &
decisions page, but the scope is confirmed only when Denis returns this block with `Confirmed by`
and `Date` completed. Until then every artifact from the page carries
`confirmation_status: unconfirmed_transcription`.

The page also prints **claim-readiness advisories** when a selected scope lacks a prerequisite, for
example a causal object without a randomized intervention, external generalization without a sixth
worm, or a binary reversal endpoint without independent video. These are not validation errors and
do not block recording; they state what the strongest claim would still need before it can be
tested.

The exported JSON includes a **scope digest**: SHA-256 over the schema and the six decisions only.
Copy that digest into the signed block so the signature is bound to the exact recorded scope. The
digest does not authenticate the signer and does not replace `Confirmed by` / `Date`; it only detects
a later edit or a mismatch between the signed brief and the transcription.

The exported JSON is validated against `schemas/scope_response.schema.json`; the schema's
`confirmation_status` is fixed to `unconfirmed_transcription`, so it cannot encode a signature.

## 6. How the returned block is checked

The Scope & decisions page is a decision room, not a free-text form. A readiness rail shows
which gate is open, each option explains what it commits the project to, and a live contract
shows the current claim ceiling, hard blockers and first action. These are derived from the
recorded answers; they are not a signature and not a scientific result.

The room also checks consistency between decisions: a causal object without a specified
target, or a BCI/NeuroAI object paired with a worm endpoint, is surfaced as a blocker rather
than silently accepted. After an export, editing any decision produces a field-by-field
before/after diff, marks the digest stage blocked, and asks for a fresh transcription before
the signature is requested. The diff covers the decisions bound by the digest, not the
provenance metadata.

Upload the returned Markdown on the Scope & decisions page. The page parses the fenced confirmation
block and reports one of these states:

| State | Meaning | What happens |
|---|---|---|
| `invalid_format` | no usable fenced block, or duplicate known fields | nothing is accepted |
| `incomplete` | a required field is empty, the date is not `YYYY-MM-DD`, or the digest is malformed | nothing is accepted |
| `no_digest` | an older block without a scope digest | cannot be bound to the transcription |
| `digest_mismatch` | the signed digest differs from the current transcription | nothing is accepted; no override |
| `blocked_on_transcription` | the six decisions or provenance are incomplete | complete the form first |
| `matches_unverified_identity` | fields are complete and the digest matches | consistency only; identity is still manual |

A matching digest proves that the returned block refers to these exact decisions. It does **not**
prove who signed it, and no upload changes the global unconfirmed status. A human must check the
reported `Confirmed by` value against the document received from Denis. The downloadable audit JSON
therefore always records `identity_verification: manual_required`.
