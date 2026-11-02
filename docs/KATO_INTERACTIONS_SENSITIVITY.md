# Bounded sensitivity and virtual perturbation

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Design

- Six named targets: AVBL, AVBR, AVEL, AVER, RIVL, RIVR. All six are present in
  every session; indices are written to `named_neuron_presence.csv`. Nothing was
  imputed.
- Three initial landmarks per session, selected by a training-only state-centroid
  rule written to `selected_sensitivity_landmarks.csv`.
- Perturbation: `+1.0` training standard deviation added to one standardized
  neuron at the initial state, then propagated for 20 frames.
- Ten permitted fits: rolling B3, frozen training B3, three chronological
  segments, five moving-block bootstrap fits.
- Two matched controls per target: nearest training variance, excluding all six
  targets. Controls are not biological sham lesions.
- Analytic Jacobian versus finite difference; the largest relative disagreement
  is `8.4e-15`, i.e. machine precision.

This is a bounded mathematical edit inside a fitted linear operator. It is not an
optogenetic, thermal, pharmacological or optogenetic-like intervention, and it
does not estimate a biological dose-response curve.

## Target summary

`response_mean` is the mean H20 response norm in standardized neuron space;
`relative_mean` divides by the baseline forecast norm.

| Target | Response mean | Relative | OOD median | OOD max | Cosine (median) | Sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AVBL | 0.579 | 0.094 | 0.010 | 0.124 | 0.441 | 0.800 |
| AVBR | 0.758 | 0.129 | 0.011 | 0.101 | 0.527 | 0.800 |
| AVEL | 1.264 | 0.208 | 0.016 | 0.100 | 0.843 | 1.000 |
| AVER | 0.709 | 0.110 | 0.015 | 0.114 | 0.603 | 0.644 |
| RIVL | 0.855 | 0.132 | 0.016 | 0.135 | 0.531 | 0.644 |
| RIVR | 1.021 | 0.164 | 0.013 | 0.131 | 0.526 | 0.533 |

Matched controls have smaller median responses (roughly `0.05-0.09`) than their
targets (`0.56-1.33`), so the selected neurons are not interchangeable with
nearby-variance controls in this fitted model. That is a **model sensitivity**
contrast, not evidence that the named neurons are causal controllers.

## Stability result

- Median target response cosine across fits: `0.556`.
- Median target sign agreement: `0.80`.
- Only AVEL has a median cosine above `0.80` and perfect sign agreement.
- Minimum pairwise cosines are negative for every target (from `-0.086` to
  `-0.81`), meaning some permitted fits reverse the direction of the response.
- The baseline forecast disagreement across fits is already large:
  median pairwise relative spread `0.785`. The instability is not created only by
  the perturbation; the permitted operators disagree about the unperturbed
  trajectory.

Gate G5 is `FAIL` under the prespecified cosine and sign thresholds.

## Empirical support

Median target OOD fraction is `0.014`, below the prespecified `0.05`; G6 is
`PASS`. But the maximum target OOD fraction reaches `0.135`, and some perturbations
leave the training 1st-99th percentile envelope in up to 13.5% of entries. A
median pass does not make every target/landmark combination interpolative.

## What can and cannot be said

**Allowed:** within the fitted B3 operator, a one-SD displacement of these named
neurons produces a nonzero, target-dependent 20-frame response; the response is
less stable across permitted fits than the target/control contrast alone would
suggest.

**Not allowed:**

- that the named neuron causally drives the response;
- that the response would occur under real stimulation;
- that any target is a better stimulation target;
- that controls are biological sham;
- that OOD support establishes validity;
- that a single fitted operator is the true network.

Every perturbation table and figure in this audit carries the exact warning:
`MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.`
Biological validation remains `NOT_EVALUABLE`.
