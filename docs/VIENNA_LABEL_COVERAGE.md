# Behaviour-label coverage and independence

Status: **partial by design of the released data**. The workbench must never turn a released label
into an independent validation label.

| Tier / source | Labels available | Frames | Independent of model? | Used for |
|---|---|---|---|---|
| BunDLe-Net release | 8 canonical behaviours | 3,005–3,295 per worm (5 worms) | Released annotation, not an external video re-annotation | Tier 1 behaviour decode/forecast targets; descriptive only |
| NC-MCM release | 8 canonical behaviours | 3,021–3,311 per worm (5 worms) | Same underlying behaviour annotation family as BunDLe-Net | Tier 2 descriptive state–behaviour MI; not independent |
| Tier 3 neural readout | Binary forward/reverse/quiet proxy | 1,401 trace frames per validation recording | Proxy derived from the paper's neural readout, not video | Internal agreement; not ground-truth behaviour |
| `reversal_events` | Binary reversal intervals | Per recording, 0-based `[start, end)` inferred | Independent release annotation | External binary check; interval/offset convention inferred |
| Full eight-behaviour video annotation | **not present** | — | — | Would be the next external validity test |

## What is missing

- No independently annotated eight-behaviour video labels for the Tier 3 recordings.
- No per-frame inter-annotator agreement for the released labels.
- No labels for a sixth worm.
- The external endpoint is binary reversal, not the full behavioural repertoire.

## Consequence for claims

- Tier 1's 83.8% behaviour accuracy is agreement with the released labels, not with an independent
  observer.
- Tier 3's AUC 0.848 is a binary reversal proxy check, not eight-behaviour validation.
- A "full behavioural decoding" claim requires new annotation with chance-corrected reliability,
  frozen splits, and a held-out worm.

## Next test

Annotate the three validation recordings for all eight behaviours with at least two blinded
annotators, report Cohen's/Fleiss' kappa, freeze the mapping, then score the existing Tier 3
readout and clamped replay without retuning. That is a data-collection task, not a code task.
