# Focused supervisor demo guide

The demo is a **single-page, read-only Streamlit entry point** for the supervisor
discussion. It is separate from the main dashboard so that the existing tabs and
the historical views are not disturbed. It reads only compact CSV/JSON/NPZ
artifacts and does **not** require model checkpoints, raw `.mat` files, or a GPU.

## 1. Launch

From the repository root (`D:\ComputaCenter\NeuroBranch\neurobranch-celegans`):

```powershell
python -m streamlit run app/handoff_demo.py --server.port 8502
```

Open <http://localhost:8502>. If port 8502 is busy, choose another port:

```powershell
python -m streamlit run app/handoff_demo.py --server.port 8503
```

The app can also be opened from the project’s existing Python environment:

```powershell
streamlit run app/handoff_demo.py
```

## 2. What the demo reads

| Artifact | Use | App dependency |
| --- | --- | --- |
| `reports/simulator/supervisor_handoff/summary.json` | headline numbers, gates, statuses | required |
| `reports/simulator/supervisor_handoff/phase_timeline.csv` | chronological account | required |
| `reports/simulator/joint_dynamics/example_observed.csv` | real PC1/PC2 continuation | required |
| `reports/simulator/joint_dynamics/example_bands.csv` | ensemble quantiles | required |
| `reports/simulator/joint_dynamics/trajectory_examples.npz` | 10-member visualization subset | optional |
| `reports/simulator/supervisor_handoff/claim_ledger.csv` | talk reference | not loaded by the app |
| `reports/simulator/joint_dynamics/score_summary.csv` | score provenance | summarized in `summary.json` |
| `reports/simulator/joint_dynamics/paired_scores.csv` | interval provenance | summarized in `summary.json` |
| `reports/simulator/joint_dynamics/calibration_h20.csv` | coverage provenance | summarized in `summary.json` |
| `reports/simulator/joint_dynamics/gates.json` | gate provenance | summarized in `summary.json` |

If the optional NPZ is absent (for example, a shallow clone without the compact
visualization subset), the trajectory section shows a notice and uses the CSV
bands instead of failing. No checkpoint is loaded.

## 3. Eight-minute route

Use the sidebar “Presentation section” radio. The route is designed to follow the
talk, not to exhaust the interface.

### 0:00–1:00 — Start here

- Read the three labels: `exploratory repeated-session reanalysis`,
  `PROBABILISTIC CALIBRATION NOT ESTABLISHED`, and
  `MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED`.
- State the provisional brief and the exact data: five Kato `WT_NoStim`
  sessions, 175 H20 landmarks, fixed 10-D PCA.
- Do not begin with the model architecture. Begin with the claim boundary.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
>
> Biological validation: `NOT_EVALUABLE`. This demo is not a systematic
> review.

### 1:00–2:30 — Timeline

- Walk the workstreams in order: representation audit, Kato baselines, Randi
  intervention, branching, simulator, external pilot, adaptive/probabilistic
  calibration, joint dynamics.
- Point out that the negative results are connected, not independent failures.
- Say explicitly: “five sessions are not five animals.”

### 2:30–4:15 — Central result

- Show the forest plot for J5 vs J1 and J5 vs J0.
- Read the two headline intervals: energy +0.446 [0.183, 0.660], trajectory
  +0.375 [0.218, 0.522].
- Point to J2: “the same mean advantage appears with a parametric diagonal
  innovation model; this is not one bootstrap.”
- Point to the caveats: reused test split, overlapping landmarks, session 1
  negative, block-10 persistence interval crosses zero.

### 4:15–5:30 — Real trajectory and mean vs stochastic

- Show the observed continuation and the J5 conditional mean in PC1.
- Compare the J1 and J5 mean paths to show the difference in trajectory shape.
- Point to the ensemble fan. Say: “the fan is the model’s uncertainty, not a
  calibrated distribution of biological futures.”
- If the NPZ is missing, use the CSV band view and say so.

### 5:30–6:30 — Calibration

- Show nominal vs empirical coverage. The 50% bar is the failure: 0.588 full
  neuron and 0.546 in the 10-D PCA.
- Show covariance error 0.604 against the 0.50 gate.
- Say: “the mean can be useful while the distribution is not calibrated.”
- Do not show a calibration plot without the gate label.

### 6:30–7:30 — Perturbation boundary

- Show the model-edit types and the explicit statement that they are state edits.
- Show that no perturbation was launched in the joint study:
  `NOT_LAUNCHED`, because calibration/covariance failed and no matched
  intervention exists.
- State: “I am not showing a perturbation trajectory as a biological prediction.”
- Mention the corrected random-control result from the historical intervention
  scan: candidate p ≈ 0.21, not specific.

### 7:30–8:00 — Decision

- Close on the two proposals: A replication, B intervention-aware design.
- Ask the six scoping questions from
  [HANDOFF_QUESTIONS_FOR_DENIS.md](HANDOFF_QUESTIONS_FOR_DENIS.md), or the
  full list if time allows.
- End with the one allowed positive sentence, not with “digital twin”.

## 4. What not to do in the demo

- Do not start with the dashboard’s 56 tabs. The focused entry point exists
  because the main app is an audit surface, not a talk.
- Do not call the ensemble a posterior over biological futures.
- Do not show a named-neuron perturbation as a predicted experimental result.
- Do not call the 10-D PCA “the latent manifold”; it is an evaluation
  coordinate system.
- Do not say “the model predicts the worm”. Say “the population mean predicts
  joint held-out trajectories better than independent AR in this representation”.
- Do not hide a failed gate. The failed calibration is part of the result.

## 5. Troubleshooting

| Symptom | Check |
| --- | --- |
| `ModuleNotFoundError: streamlit` | Activate the project environment; install from `pyproject.toml` if needed |
| `FileNotFoundError` for summary | Run `python scripts/build_supervisor_handoff.py` first |
| Trajectory section shows a notice | The optional NPZ is absent; the CSV view is sufficient |
| Port already in use | Use `--server.port 8503` |
| Plotly chart does not render | Use a current browser; Plotly is a project dependency |
| App starts but is empty | Confirm the working directory is the repository root |

## 6. Demo evidence status

The demo is a presentation aid, not a new experiment. Every number in it is
loaded from the frozen reports. If a number and the report disagree, the report
wins. The validator `python scripts/validate_handoff_reports.py` checks the
compact summary and manifest.
