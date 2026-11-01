# Oxygen input audit and models

## 1. Data audit

Source: Kato et al. (2015) `WT_Stim.mat`, seven sessions, external to Git.

- SHA256: `96ffc39afbc90b66ff1b4c98de199f8424391450d4a8050cbe2b65b3a2fdaa72`
- 12 frame-level oxygen transitions per session.
- Two measured levels: 4% and 21%.
- The first 360 s is constant 21%; the schedule then toggles at nominal 30 s intervals.
- The **actual** transition frames differ slightly by session (e.g. first transition 360.000 s in some sessions, 360.178 s in others). The audit records actual frame times; no exact 30 s frame alignment is claimed.

Default chronological split for `kato_stim_0`:

| Split | Frames | Oxygen transitions inside |
|---|---:|---:|
| train | 1465 | 4 |
| validation | 366 | 4 |
| test | 343 | 3 |

Across all seven sessions, 21 held-out transition frames are available. The same fixed schedule is repeated in every session, so these are not 21 independent biological events. This is a low-power, exploratory analysis.

## 2. What oxygen is

Oxygen is a **controlled environmental sensory input**. It is not optogenetic stimulation, not a targeted neuronal perturbation, and not a controlled input to a specific neuron. The oxygen-conditioned forecasts are *known-schedule-conditioned*: the measured oxygen series for the scored window is supplied. They are not forecasts of a future stimulus and not autonomous rollouts.

## 3. Models

| Label | Model | Input |
|---|---|---|
| A | stable latent LDS, no input | none |
| B | stable latent LDS | measured O2 |
| C | compositional latent LDS | measured O2, independent coefficients |
| D1 | stable latent LDS | O2 delayed 12 frames (~4 s) |
| D2 | stable latent LDS | elapsed time since last O2 transition |
| D3 | stable latent LDS | linear time index |
| E | stimulus-only baseline | current O2, no neural history |

H20 is evaluated on an evenly spaced subset of up to 40 held-out starts for every model. H100 is not used in this low-power ablation; it is covered in Stage B. The subset size is recorded in `oxygen_model_comparison.csv`.

## 4. Results

See:

- `oxygen_model_comparison.csv` — per session/horizon metrics;
- `oxygen_model_aggregate.csv` — mean/median skill;
- `oxygen_pairwise.csv` — paired bootstrap of B−A and C−B;
- `oxygen_one_step.csv` — one-step metrics;
- `oxygen_model_summary.json` — semantics and power warning.

Results are inserted in `_RESULTS.md` after the script completes. Positive or negative, the interpretation is constrained by seven sessions, a shared schedule, and three held-out transitions per session.

## 5. Reproduction

```powershell
python scripts/context_oxygen_audit.py
python scripts/context_oxygen_models.py
```

The external `.mat` file is not downloaded automatically by these commands. Set `TEMP`/the path if the file is elsewhere; the script fails explicitly if the file is absent.
