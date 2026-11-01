# Oxygen stimulus response — is there a reproducible evoked response?

Artifacts: `reports/simulator/failure_diagnostics/oxygen/`
Data: Kato 2015 WT_Stim, seven sessions, 4% / 21% oxygen.
Code: `src/neurobranch/diagnostics/oxygen.py`, `scripts/genfail_oxygen.py`

## 1. What the stimulus actually is

Oxygen is an **environmental sensory input**, not targeted optogenetic or cellular stimulation. The recordings contain a repeated schedule of oxygen transitions. There are 12 transitions per session, from 360 s to 690 s, alternating 21% → 4% (“down”) and 4% → 21% (“up”), with a 30 s period and a small frame-rate jitter (SD ≤ 0.11 s across sessions).

The seven sessions are repeated recordings of the same schedule, not seven independent animals and not 84 independent events. Test evaluation uses only the last three transitions per session (21 events total). This is severely underpowered for a response-shape claim.

## 2. Verified event facts

| property | value |
|---|---|
| sessions | 7 |
| transitions per session | 12 |
| first transition | 360.00–360.18 s |
| last transition | 690.01–690.23 s |
| directions | 42 down, 42 up |
| sampling | ~3 fps |
| named sensory neurons present | AQR, BAGL, BAGR, URXL, URXR in at least one session |

Not every sensory neuron is present in every session. For example, AQR is absent in several sessions, and URXL/URXR are absent in others. A sensory-neuron-specific claim cannot be made uniformly across sessions.

## 3. Response definitions

- **Baseline:** causal pre-event mean over the preceding 5 s.
- **Response vector:** post-event deviation from that baseline.
- **Latency:** peak absolute deviation within 10 s.
- **Reliability:** correlation between response vectors of repeated events in the same session.
- **Direction contrast:** cosine between the mean up-response and mean down-response.

All baselines use only past frames. No future-dependent smoothing was used.

## 4. Is the response reproducible?

Across-event response-vector correlations are near zero:

| session | mean event-pair correlation | median |
|---|---:|---:|
| kato_stim_0 | 0.020 | 0.003 |
| kato_stim_1 | 0.037 | 0.016 |
| kato_stim_2 | 0.041 | 0.036 |
| kato_stim_3 | 0.002 | −0.063 |
| kato_stim_4 | −0.005 | −0.004 |
| kato_stim_5 | −0.015 | −0.025 |
| kato_stim_6 | −0.011 | 0.000 |

There is no consistent shared response vector across repeated transitions. Direction contrast is also weak and inconsistent: up-minus-down cosine ranges from −0.34 to +0.18 across sessions, with no stable sign. Mean up and down response magnitudes are small relative to trace variability.

## 5. Does a response model beat controls?

### 5.1 Absolute-X model (diagnostic, drift-dominated)

This model predicts the absolute standardized trace. It is included because it is the naive comparison, and it fails:

| model | mean skill vs persistence |
|---|---:|
| placebo event | −3.982 |
| event only | −4.031 |
| time only | −19.368 |
| placebo + time | −19.408 |
| event + time | −19.443 |

Large slow fluorescence drift dominates absolute prediction. This table is not evidence for or against a response; it shows why absolute-X fitting is a poor test.

### 5.2 Baseline-corrected event-window model

Targets are post-event deviations from the causal pre-event mean. Controls use event-relative elapsed time (linear + quadratic), event-lag one-hot, or placebo events 15 s after the real ones. Skill is versus a no-response prediction of zero deviation.

| model | mean skill vs no response | median |
|---|---:|---:|
| time only | −0.097 | −0.106 |
| event + time | −0.184 | −0.212 |
| event lag | −0.194 | −0.221 |
| placebo event lag | −0.269 | −0.347 |

All models are worse than simply predicting no response. The event model does **not** beat the time-only control. The placebo is not better either; there is no evidence that the event kernel captures a real response.

### 5.3 Session-disjoint named-neuron model

Using 13 common named neurons and holding out one session at a time:

| model | mean skill vs no response |
|---|---:|
| event only | −6.948 |
| event + time | −10.629 |
| time only | −11.798 |

Event-only has lower absolute MSE than time-only in 6/7 held-out sessions, but all models are far worse than the no-response baseline (MSE ~0.6–0.9 versus ~0.1). The apparent event advantage is baseline/session scaling, not a validated response. With three test transitions per session and 13 neurons, this design cannot establish a sensory response.

## 6. Oxygen versus elapsed time

The schedule is **not randomized**: every session receives the same transitions at the same elapsed times. Therefore an oxygen effect and a time-since-start effect are not separately identifiable from these recordings. Quantitatively:

- the event-presence design is predicted by a low-order time basis with R² ≈ 0.02;
- the point-biserial correlation between elapsed time and event presence is ≈ 0.25;
- event lags and time basis are not pathologically collinear (condition number ≈ 26–27);
- but the design is still confounded by construction: all events are time-locked, and there is no jittered or sham schedule.

A low linear R² does not remove the confound. A flexible time model can always approximate a fixed schedule given enough basis functions; conversely, a true oxygen effect can be misattributed to elapsed time. The correct statement is **“oxygen and elapsed time are not experimentally separated”**, not “there is no time confound because R² is low”.

## 7. Answers

- **Q9:** No reproducible, measurable evoked response was detected at the population level. Event-pair correlations are near zero, direction contrasts are inconsistent, and no response model beats no-response.
- **Q10:** Oxygen effects are **not distinguishable from elapsed time** with this fixed repeated schedule.
- **Q11:** No. A simple stimulus-response model does not beat time-only controls; both are worse than no-response.

## 8. Limits and next measurement

These data cannot establish a sensory response because:

1. only 3 test transitions per session;
2. the schedule is identical across sessions, so event and time are confounded;
3. no sham/randomized transition times;
4. no independent animals;
5. calcium observation and population sampling noise;
6. sensory neuron availability varies across sessions.

A decisive next experiment would use randomized transition times, a sham schedule, more transitions per animal, and independent animals, with the same causal baseline definition. Until then, the honest verdict is `OXYGEN_RESPONSE_ASSESSED` — PASS for the assessment; the biological response is **not established**.
