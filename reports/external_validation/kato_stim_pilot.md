# Kato Stim constant-oxygen external pilot

**Warning:** EXPLORATORY EXTERNAL AUTONOMOUS-DYNAMICS PILOT — NOT A SENSORY-RESPONSE, DIGITAL-TWIN OR CAUSAL-INTERVENTION RESULT.

- Protocol: `B_per_animal_refit`; evidence tier: `exploratory_external_pilot`.
- Segment: 0 <= t < 360.0 s, O2 = 21.0% (no switch in segment).
- Fixed config: {"lag": 2, "ridge": 10.0, "spectral_cap": 0.995, "support_quantile": 0.99}; model selection: none.
- Sessions: 7/7 fitted; positive skill: 0.
- Median skill: -0.6025; mean skill: -0.5933; bootstrap 95% CI: [-1.02403218480517, -0.24116359993628472].

| Worm | Traces used | Test starts | Spectral radius | H20 MSE | Persistence MSE | Skill | Energy | OOD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 102 | 30 | 1.0369 | 0.6398 | 0.6180 | -0.0353 | 5.022 | 0.0000 |
| 1 | 117 | 30 | 0.9951 | 1.6909 | 0.5984 | -1.8258 | 12.493 | 0.0444 |
| 2 | 121 | 30 | 1.0391 | 1.4635 | 0.8563 | -0.7091 | 11.531 | 0.0444 |
| 3 | 129 | 30 | 1.0652 | 1.6674 | 1.0173 | -0.6390 | 11.798 | 0.0000 |
| 4 | 118 | 30 | 0.9969 | 0.8993 | 0.7981 | -0.1268 | 4.830 | 0.0000 |
| 5 | 145 | 30 | 0.9906 | 1.1169 | 0.9197 | -0.2144 | 5.859 | 0.0127 |
| 6 | 140 | 30 | 1.0038 | 1.2309 | 0.7681 | -0.6025 | 9.092 | 0.0016 |

## Interpretation

No external animal beat persistence at H=20. The result is a negative replication of the Phase 6 conclusion, not a sensory-response test: oxygen was constant and no exogenous input was supplied. The seven sessions are not proven independent animals, and the cohort is small. Frozen Phase 5/6 VAR transfer is blocked by worm-specific 103–127-dimensional trace panels; only per-animal refits are meaningful. A failure here does not establish that the model cannot work under an input-aware design; it does establish that the current autonomous pipeline does not transfer predictively to this constant-oxygen external segment.

Files: `kato_stim_pilot_summary.json`, `kato_stim_pilot_per_worm.csv`, `kato_stim_pilot_starts.csv`.
