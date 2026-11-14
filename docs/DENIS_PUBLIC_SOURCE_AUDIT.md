# Denis Kovačević — public-source audit

**Status:** public-record audit, last checked **2026-11-13**. This is not a Denis-specific
requirements document, not a systematic review, and not evidence that Denis endorses the studio.
It records what the group's public pages do and do not say so the product stops treating
group-level alignment as a personal specification.

## 1. Pages checked

| Source | URL | What it establishes |
|---|---|---|
| Group team page | <https://ni.cs.univie.ac.at/about-us/team/> | Denis Kovacevic is listed as research and teaching staff, BSc M.Sc., Research Group Neuroinformatics, room 02.45. No research statement or publication list on the page. |
| UCris person profile | <https://ucrisportal.univie.ac.at/en/persons/denis-kovacevic/> | Person record exists; no publications are listed in the profile section checked. |
| Group projects | <https://ni.cs.univie.ac.at/research/projects> | *An interdisciplinary approach to learn and test the causal mapping between neural network dynamics and behavior*, 2024-07-01 to 2028-06-30, Zimmer lead, Grosse-Wentrup and Tchumatchenko co-leads. |
| Group open topics | <https://ni.cs.univie.ac.at/teaching/open-topics> | Group-level teaching topics: causality and causal inference; machine learning and causal modeling in cognitive neuroscience; BCI for communication and rehabilitation. |

The person URL formerly used by the studio,
`https://ni.cs.univie.ac.at/about-us/team/person/unitid/81/denis-kovacevic`, returned HTTP 404
without a `cHash` query parameter. A variant with the team page's `cHash` returned HTTP 200, but
that parameter is not a stable citation target. The studio now links to the stable team page and
the UCris profile instead.

## 2. What the public record supports

- Denis is a member of the Research Group Neuroinformatics at the University of Vienna.
- The group has an active causal-mapping project running from July 2024 to June 2028.
- The group advertises work in causality, causal modeling in cognitive neuroscience, and BCI.
- The group's published papers used by this studio are BunDLe-Net (preprint), NC-MCM
  (peer-reviewed), and the Morrison–Young premotor model (peer-reviewed).
- The studio's three tiers are aligned with those outputs at the level of method and vocabulary.

## 3. What the public record does not support

- There is no public Denis-specific research statement, thesis abstract, project brief or
  supervision agreement in the pages checked.
- No publication is listed on the checked group or UCris pages. This is not proof that no
  publication exists elsewhere; it only means the checked pages do not list one.
- The public record does not say whether Denis wants a causal experiment, external replication,
  a reusable workbench, or a BCI/NeuroAI lane.
- It does not say which behavioural endpoint, dataset, intervention, success criterion, stop rule
  or budget he wants.
- It does not attribute BunDLe-Net or NC-MCM to Denis. Those papers remain attributed to their
  authors.

## 4. Required next evidence

The only artifact that can convert this audit into a specification is Denis's written answer to
the six decisions in [`DENIS_REQUIREMENTS_BRIEF.md`](DENIS_REQUIREMENTS_BRIEF.md). Record a
transcription on the **Scope & decisions** page; that transcription is not a signature. The signed
confirmation block in the brief remains authoritative.

If Denis supplies a publication, thesis, grant text or supervision note, add it here with the exact
URL or file hash, the date checked, and a one-line statement of what it establishes. Do not infer a
requirement from a group-level page.
