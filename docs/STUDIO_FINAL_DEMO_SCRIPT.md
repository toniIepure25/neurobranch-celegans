# Final supervisor demonstration script

**Target:** 8–10 minutes, seven Studio sections plus presentation mode.  
**Mode:** local, read-only. No live fitting.

## 1. Preflight

```powershell
cd D:\ComputaCenter\NeuroBranch\neurobranch-celegans
python -X utf8 scripts/studio_release_checks.py
python -X utf8 scripts/studio_provenance_audit.py --pretty
python -X utf8 -m streamlit run app/neurobranch_studio.py
```

The provenance command is expected to exit `1` while the two historical
audit-script snapshots remain unverifiable. Do not hide that; state it if asked.

Offline fallback:

- use `reports/scientific_studio/screenshots/` and the contact sheets in
  `reports/scientific_studio/release_candidate/`;
- open `docs/STUDIO_FINAL_DEMO_SCRIPT.md` and the evidence ledger;
- do not refit a model during the demonstration.

Single evidence ledger: [CLAIMS.md](CLAIMS.md).

## 2. Two-page scientific summary

**Question.** Does a full population VAR retain H20 predictive structure beyond
a ten-factor common-state control in five Kato 2015 `WT_NoStim` recordings?

**Answer.** B3 has lower H20 PCA-10 energy than B5-10 by `+0.2644` (B5−B3
convention), about `9.79%` of the B3 mean. The conditional landmark interval is
`[+0.0750, +0.4031]`. The recording-resampling interval is
`[-0.0408, +0.4793]` and includes zero. Four of five sessions are positive; the
sign test is not significant. H1 reverses. MSE improvement is not established.

**Why it is exploratory.** The test tail was reused; the five recordings are not
documented independent animals; B3 and B5 have different capacity; calibration
is marginal, not joint; the fitted operators are unstable; the result cannot
identify direct biological coupling.

**What is not claimed.** No mechanism, causality, digital twin, animal-level
generalization, calibrated biological uncertainty, or stimulation target.

## 3. Eight-to-ten-minute route

### 0:00–1:00 — Motivation and unit

Open **Overview**. Say:

> “Five recordings, 175 overlapping landmarks. The landmark is the unit of
> computation, not an animal. I am going to show a bounded methodological
> result, not a validated simulator.”

Point to the archive provenance and the exploratory badge.

### 1:00–2:00 — Neural activity

Open **Neural Activity**. Show raw ΔF/F and the standardized trace. Explain that
the models consume the train-standardized series and that the selected origin is
shared with the 3D view. Do not interpret the trace as a circuit.

### 2:00–3:30 — 3D trajectories

Open **3D Neural Dynamics**. Rotate once, press Play briefly, stop.

> “PC1–PC3 are 52% of total training variance, which is 73% of the retained
> PCA-10 variance. The other 27% of PCA-10 is not shown. A point is a
> population-state summary, not anatomy. Cyan is measured; amber is B3; violet
> is B5; the dotted continuation is held-out measurement used only for display.”

If WebGL fails, switch to the 2D fallback and say so.

### 3:30–4:30 — Model equations

Open **Models & Forecasting**. Show B0 persistence, B1m independent AR, B3
population VAR, B5 own AR plus ten train-only factors. State:

> “B3 has 7.96–9.95× the nominal coefficient count of B5-10. This is not a
> capacity-matched comparison.”

### 4:30–5:30 — B3 versus B5 and uncertainty

Open **Uncertainty & Evidence**. Show `+0.2644`, the conditional interval, and
the recording interval crossing zero. Explain that the conditional interval
answers “which landmark?” and the recording interval answers “which recording?”
They are not interchangeable. Show the MSE null and the H1 reversal.

### 5:30–6:30 — Negative findings

Stay in Evidence or open **Research Notebook**. State:

- mean-MSE improvement not established;
- marginal coverage not joint calibration;
- operators unstable;
- residual mechanism unidentified;
- capacity confound unresolved;
- test tail reused.

### 6:30–7:30 — Virtual perturbation

Open **Virtual Perturbations**. Apply a `+1.0` training-z edit at a saved
landmark. Read the exact warning:

`MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.`

Explain the first-step response is the fitted `A` column; B3/B5 disagreement is
model dependence; the lab does not stimulate, rank targets, or predict behavior.

### 7:30–8:30 — Next steps

Open **Research Notebook** and the next-experiment card:

1. independent animals with documented identity;
2. pre-registered H20 endpoint and baseline family;
3. capacity-matched residual model;
4. sham-controlled perturbation;
5. a clean test set not used for iteration.

### 8:30–9:00 — Close

> “The defensible claim is a within-recording, landmark-level, H20 exploratory
> contrast under a training-only PCA. The recording-level interval includes
> zero. We are ready for methodological review, not confirmatory or biological
> claims.”

## 4. Likely supervisor questions

| Question | Short answer |
| --- | --- |
| Is this five animals? | No. Five recordings; animal identity is inferred, not documented. |
| Why does the interval cross zero? | The recording-resampling interval treats recordings as the unit; there are only five. |
| Is B3 simply bigger? | Yes, capacity is 7.96–9.95×; the contrast is not capacity-matched. |
| Does lower energy mean better forecasts? | Not necessarily; MSE improvement is not established. |
| Is H20 cherry-picked? | H1 reverses and the endpoint is exploratory; a future study must pre-register it. |
| Can you stimulate a neuron? | No. The perturbation is a mathematical counterfactual. |
| Is this a digital twin? | No. |

## 5. Failure protocol

Stop the live demo and use screenshots if:

- the 3D route is blank and the 2D fallback fails;
- a displayed number differs from Evidence;
- the perturbation warning is missing;
- the app requires network access;
- a question demands a causal or animal-level claim the evidence cannot support.

Do not fit a new model or substitute a placeholder result.
