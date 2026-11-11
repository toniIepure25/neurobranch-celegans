# Scope: BCI, EEG, fMRI and MeDIL

Status: **not implemented in v2**. This is a scope statement, not a claim of
integration. The studio is a *C. elegans* whole-brain / behaviour workbench built
from the Vienna group's published BunDLe-Net, NC-MCM and premotor artifacts.

## In scope now

- 3-D BunDLe-Net latent trajectories and held-out-worm open-loop forecasts.
- NC-MCM descriptive state model, assumption audit, order-2 predictive follow-up,
  transport and K-sensitivity checks.
- Published 84/15/69 clamped premotor simulation, Python/Julia parity, β=0 and
  zero-signal controls, external binary-reversal audit.
- Provenance, claim ledger, evidence report and lab handoff.

## Not in scope now

| area | why it is absent | what would be required |
|---|---|---|
| EEG/MEG/ECoG BCI | no human neural data, no acquisition hardware, no BIDS modality pipeline | ethics/consent, hardware, BIDS EEG/MEG/iEEG ingestion, artifact rejection, leakage-safe decoding |
| fMRI | no MRI data or preprocessing | BIDS MRI/fMRI, motion/censoring, GLM or encoding model, subject-level splits |
| MeDIL | no MeDIL dataset or causal-discovery implementation in this release | released data, identifiability assumptions, validated causal-discovery baseline |
| closed-loop BCI | no real-time acquisition or stimulation | latency budget, safety review, online decoder, sham/closed-loop controls |
| digital twin | no individual calibration or intervention validation | per-animal parameters, perturbation data, prospective validation |

## Adapter boundary

If a future release adds one of these modalities, it must enter through a new
loader and a new evaluation contract; it must not be presented as an extension of
the worm results. Shared vocabulary ("latent", "state", "decoder") is not shared
evidence.

## Claim rule

No UI page, report or product brief may say "BCI-ready", "EEG-compatible",
"fMRI-validated", "MeDIL-integrated" or "digital twin" for v2. The honest label is
**C. elegans research prototype, lab-review-ready**.
