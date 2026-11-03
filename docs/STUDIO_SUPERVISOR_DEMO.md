# Supervisor demonstration for Denis Kovačević

**Duration:** 8 minutes, plus questions.
**Mode:** `Presentation mode` for the first six minutes, then the live 3D
explorer and evidence tabs.
**Goal:** leave with a defensible exploratory result, an explicit unit of
inference, and a falsifiable next experiment — not with a claim that the model
is a digital twin.

## Before the meeting

```powershell
cd <repository root>
python -X utf8 -m streamlit run app/neurobranch_studio.py
```

Check:

- sidebar badge reads `EXPLORATORY`;
- Overview shows 175 landmarks, 5 recordings, +0.2644, 9.79%;
- 3D page renders;
- `Presentation mode` opens slide 1;
- the browser is at 1920×1080 or 1366×768.

## Slide-by-slide script

### 1. A bounded result is still a result

> "This is a reanalysis of five Kato 2015 WT_NoStim recordings. The question is
> narrow: does a full population VAR retain predictive structure beyond a
> ten-factor common-state control at a 20-step energy endpoint? The answer is
> exploratory, horizon-dependent, and bounded by five recordings. I want to
> show you what it does not establish as clearly as what it does."

Do not open with the 3D plot. Open with the unit.

### 2. The unit is a recording and an overlapping landmark

> "There are 175 evaluation landmarks. They are overlapping windows, not 175
> independent animals. The effective number of non-overlapping H100 proxies is
> about 25. Four of five recording means are positive; with five units the
> recording interval includes zero. So the honest object is a within-recording,
> landmark-level contrast."

### 3. A state space, not an anatomy

Open the 3D explorer. Rotate once; press Play briefly; stop.

> "These axes are PC1–PC3 of a training-only 10-component PCA. The first three
> explain about 52% of training variance. A point is a population state summary.
> It is not a worm brain, a connectome, or a physical space. Cyan is measured;
> dotted cyan after the diamond is held-out measurement shown only for
> evaluation; amber and violet are model forecasts from the same origin."

### 4. Four models, one contrast

Go to **Models & Forecasting**.

> "B3 is the full 121-neuron population VAR. B5 is each neuron's own lag plus
> ten common factors. B1m is an independent AR with the historical 200-frame
> window. B0 is persistence. The primary contrast is B5 minus B3 at H20; a
> positive value means B3 has lower energy. We are not selecting among these on
> the test tail."

### 5. The result, with both intervals

Go to **Uncertainty & Evidence → Primary contrast**.

> "The weighted mean is plus 0.264, about 9.8% of the B3 mean. The conditional
> landmark interval is plus 0.075 to plus 0.403 and excludes zero. But the
> recording-resampling interval is minus 0.041 to plus 0.479. The five-recording
> t interval is minus 0.064 to plus 0.587, and the sign test is p=0.1875. So the
> result is exploratory. It is not a confirmed population effect."

Point to the forest, then the bootstrap table. Do not call the conditional
interval "the" confidence interval.

### 6. The boundary is not one number

Go to **Horizon & proper scores**, then **Calibration**, then **Capacity**.

> "At H1 the sign reverses: minus 0.064. The advantage appears at H5, H10, H20.
> MSE is null: plus 0.055 with an interval crossing zero. B5's 95% marginal
> coverage is 0.725; B3's is 0.938, but that is marginal, not joint, calibration.
> B3 has 7.96 to 9.95 times the nominal parameters. The residual advantage is
> not identifiable as cross-neuron information. The 51/49 split is an ordered
> arithmetic decomposition, not a unique structural partition."

### 7. What the perturbation lab does not do

Optional, 40 seconds.

> "This edits one standardized calcium coordinate in a fitted model and rolls
> forward. It is a model counterfactual. It is not optogenetics, not a
> stimulation, and it does not rank targets. The B3/B5 disagreement shows model
> dependence; agreement would not validate it."

### 8. Next experiment and falsification

Go to **Research Notebook → Next experiments**.

> "The next study needs independent animals with a primary animal key, a
> capacity-matched residual model or a measured common input, a fresh test tail,
> and joint calibration. The falsifier is simple: if the animal-level
> B3-minus-B1 energy interval includes zero, or the sign reverses in most
> animals, this residual story does not survive."

Close:

> "The deliverable is a reproducible instrument for reading a bounded result,
> plus a clear list of what would make it stronger. It is not a validated digital
> twin."

## Anticipated questions

**"Is B3 better?"**
At H20 PCA-10 energy in these five recordings, on average yes by +0.264; at H1
it is worse; MSE is null; the recording-level interval includes zero. The
defensible word is "exploratory".

**"Why not just use more landmarks?"**
They overlap. More overlapping windows do not add independent animals. The
recording is the resampling unit for generalization.

**"Is the 3D plot a brain?"**
No. It is a PCA projection. Distance is statistical, not anatomical.

**"Can you read connectivity from the VAR?"**
No. The coefficients are effective lag-one transitions on standardized calcium
signals at imaging timescale. They are not synapses or directed connections.

**"Is the ensemble calibrated?"**
Not jointly. B5's 95% marginal coverage is 0.725. The Studio shows marginal
coverage and labels it as such.

**"Is this a digital twin?"**
No. It is a read-only evidence layer and a model counterfactual lab. There is
no validated intervention, no closed loop, and no biological effect size.

**"What is the single strongest number?"**
The primary paired mean +0.2644 with the conditional block-5 interval
[+0.0750, +0.4031]. The strongest *honest* sentence includes the recording
interval and the H1 reversal.

## If something fails

- 3D blank: switch to **2D fallback** and continue; do not debug in the meeting.
- App error: the error names the artifact; use the committed screenshots in
  `reports/scientific_studio/screenshots/`.
- Presentation route: **Exit presentation** returns to the Studio.
- Legacy demo: run `app/handoff_demo.py` separately; it is not the primary
  surface.
