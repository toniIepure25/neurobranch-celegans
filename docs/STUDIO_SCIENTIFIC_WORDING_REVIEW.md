# Scientific wording review

**Scope:** supervisor-facing Studio text, presentation slides, sidebar, footer,
and active error/status messages.  
**Out of scope:** frozen manifests, historical errata rows, and numerical
artifacts. Nothing in this review changes a calculation.

## 1. Required replacements

### 1.1 “Prespecified” contrast

The old active wording implied a prespecified or confirmatory contrast. The
reported analysis does not support that status. The active text now reads:

> The reported contrast is energy(B5) − energy(B3); positive values favor B3.
> The sign convention was clarified during the exploratory analysis.

Use this sentence in the Overview and in any spoken framing. Do not call the
contrast prespecified, preregistered, or confirmatory.

The future-experiment slide may use “prespecified” only for a proposed design
target, dose, time window, and sham/control. That is a design requirement for a
new study, not a description of the completed analysis.

### 1.2 Test-tail optimization

The old sentence “No test-tail performance was optimized” was too broad: earlier
research phases reused the evaluation tails. The active text now reads:

> No additional test-tail model optimization was performed for this release.
> Earlier research phases reused the evaluation tails; findings remain
> exploratory.

This is the only defensible release-scope statement. It does not claim that no
test-tail contact ever occurred.

### 1.3 Unit of inference

The old phrase “five preparations” invited the reading that five independent
animals were recorded, or that one animal was recorded five times. Neither is
established. Active text uses:

> Five recordings from one experimental cohort.

The animal mapping is inferred and is not an authoritative identity key. Do not
say “five animals”, “five worms”, or “five independent replicates”.

## 2. Where the changes were applied

| Surface | Change |
| --- | --- |
| Overview lead | Reported-contrast sentence and sign-convention clarification. |
| Overview question | “beyond a ten-factor common-state control at H20?” |
| Overview evidence panel | One prominent `EXPLORATORY` status; unit, H1 reversal, MSE, calibration, capacity, and release scope in supporting text. |
| Overview limitation panel | One experimental cohort; inferred animal mapping. |
| Slide 01 | “Five recordings, one experimental cohort, and 592 included predictors.” |
| Slide 02 | Three KPI cards; provenance and cohort-reading panels; no preparation count. |
| Slide 04 | “fitted model family” and “reported contrast”; no prespecified contrast. |
| Slide 05 | Direction and zero-crossing caveat in the one-line message; no confirmatory framing. |
| Slide 08 | “Can defend / Cannot defend” separation. |
| Slide 09–10 | Falsifiers and a new-study design; no completed mechanism claim. |
| Sidebar | `5 RECORDINGS`, `1 EXPERIMENTAL COHORT`, `EXPLORATORY`. |
| Footer | Release-scope sentence with earlier tail reuse. |
| Evidence section | One experimental cohort; inferred animal mapping; release-scope sentence. |
| Models section | “fitted model family”; “reported contrast”. |
| Errata X02/X15 active notes | “five recordings from one experimental cohort”. |
| Metadata docstring | “reported” instead of “prespecified”. |

## 3. Scientific status that must remain visible

- The analysis is **exploratory**.
- The evaluation tails were reused in earlier research phases.
- The primary contrast is **H20 PCA-10 energy**: B5−B3 =
  `+0.2644379926683563`.
- The conditional landmark interval is
  `[+0.0749929196, +0.4031164916]`.
- The recording-resampling interval is
  `[-0.0408009083, +0.4792966063]` and includes zero.
- 4 of 5 recordings are positive; the sign-test p-value is `0.1875`.
- The H1 contrast reverses (`-0.0636694611`); the effect is horizon-dependent.
- The MSE difference is `+0.0545043099`, interval
  `[-0.1574261907, +0.2409040856]`; an MSE advantage is not established.
- B3 and B5 capacity is unmatched (`7.9583952452–9.9485924113×`).
- Coverage differs (B3 `0.9382857143`; B5 `0.7245714286`).
- The 3D view is a training-only PCA projection, not anatomy.
- The perturbation lab is model-dependent and not causally validated.

## 4. Deliberate residual wording

The Research Notebook still displays historical errata, including the exact
language that was corrected. Those rows are marked historical and are not active
claims. The presentation presenter note also says “do not say five preparations”
as an instruction; the phrase is not used as a claim.

Searches for `prespecified` therefore find two legitimate residuals:

1. the future-experiment design requirement on slide 10;
2. the presenter-note prohibition in slide 2.

A search for `preparation` may also find the explicit correction note. These are
not active cohort descriptions.

## 5. Review checks

- [ ] No active surface calls the completed contrast prespecified or
      confirmatory.
- [ ] No active surface says no test-tail contact ever occurred.
- [ ] No active surface says five animals, five worms, or five independent
      replicates.
- [ ] The reported contrast sign convention is stated before interpretation.
- [ ] The recording interval crossing zero is visible wherever the primary
      number is interpreted.
- [ ] The capacity mismatch and calibration difference are not footnotes only.
- [ ] No causal, connectome, digital-twin, or anatomical claim is implied.

Automated regression coverage:

- `tests/test_studio_scientific_wording.py`
- `tests/test_studio_app_smoke.py::test_slide_two_metrics_and_cohort_wording`
