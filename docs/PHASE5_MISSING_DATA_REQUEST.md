# Draft data/code request to the Kramer et al. authors

**Status:** local draft only. Not sent.

**Subject:** Request for aligned SAAV decoding inputs and preprocessing metadata

Dear Dr Kramer and colleagues,

We are independently rebuilding the SAAV post-reversal turn-direction decoding
analysis in the NeuroBranch project. We have read the paper, the bioRxiv
preprint and the public `flavell-lab/SAAV_Decoding` repository (commit
`c7e7e18a84cde5dc02df9d63492293b38402eb2c`). We are not asking you to endorse
our results; we want to distinguish a true methodological difference from an
artefact of our reconstruction.

The public `load_data.py` reads four files from an author-local path:

```text
/store1/prj_rim/Decoding_Data/FlossieAligned/SAAV_dorsal.csv
/store1/prj_rim/Decoding_Data/FlossieAligned/HC_dorsal.csv
/store1/prj_rim/Decoding_Data/FlossieAligned/SAAV_ventral.csv
/store1/prj_rim/Decoding_Data/FlossieAligned/HC_ventral.csv
```

Could you share these files, or the script and metadata needed to regenerate
them from the Dryad HDF5 recordings? Specifically:

1. The four aligned CSVs, or a private archive with the same contents.
2. The preprocessing script that generated them from raw traces and behaviour.
3. Column definitions: event order, frame count, SAAV and head-curvature
   channels, units and sign convention.
4. Event inclusion rules: minimum reversal length, complete-case/NaN policy,
   near-zero post-turn exclusion, and how many events were excluded at each
   step.
5. Head-curvature definition and alignment procedure: smoothing, zero-crossing
   or phase interpolation, samples per head-swing cycle, frame count, and how
   the reversal end is placed in the window.
6. Z-score fit scope: per recording, per animal, per split or global; whether
   future frames contribute.
7. Animal, recording, cohort and event identifiers for every row, so that the
   same events can be evaluated with animal-disjoint splits.
8. Original fold assignments, random seeds, and per-event/per-section
   predictions if available.
9. The exact software environment and any post-publication code changes.

We would also value a pointer to the analysis code that produced Extended Data
Fig. 2d, if it differs from the public notebook, and clarification of whether
the reported test accuracy is computed per four-frame section (as the current
`train.py`/`evaluate` path suggests) or per event.

We are happy to acknowledge the source of the data and to share our
reconstruction audit and any discrepancy we find. We will not redistribute
restricted data without permission.

Thank you for considering this request.

Sincerely,  
The NeuroBranch project
