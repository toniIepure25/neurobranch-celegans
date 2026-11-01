# Latent models and corrected VAR

## 1. Model

`StableLatentLDS`:

- PCA fitted on the **training frames only**;
- latent transition `z_{t+1} = A z_t + b + B u_t + noise`, fitted by ridge on training transitions;
- `A` capped with the exact latent spectral-radius helper;
- mean-preserving intercept when requested;
- Ledoit–Wolf process covariance and diagonal observation covariance;
- causal Kalman-style filter for task B;
- deterministic and process-covariance rollouts for task C;
- optional known input `u_t` for task D.

`StableVARModel`:

- unconstrained ridge VAR fit first;
- iterative companion-matrix cap;
- realized radius verified after fitting;
- new class and new name (`var_stable`); frozen Phase 6 checkpoints are untouched.

## 2. Selection

The Stage B selection rule was fixed before looking at the test block:

> highest median validation H20 skill vs persistence across the five chronological holdouts.

Selected configuration:

- latent dimension **15**
- ridge **0.01**
- spectral cap **0.995**
- stable VAR lag **1**, ridge **1.0**

Sixty latent configurations (5/10/15 dimensions × four ridge values × three caps) and twenty VAR configurations were evaluated. The test block was evaluated only after selection.

## 3. Test results (chronological final 15%)

All values are skill vs persistence. Mean/median are across five recording sessions, not five independent animals.

| Model | Horizon | Mean skill | Median skill | Mean support q99 fraction |
|---|---:|---:|---:|---:|
| latent LDS | 20 | −1.616 | −0.911 | 0.988 |
| latent LDS | 100 | −0.715 | −0.164 | 0.998 |
| latent one-step | 1 | −2.408 | −2.428 | — |
| stable VAR | 20 | −1.481 | −0.770 | 0.988 |
| stable VAR | 100 | −0.612 | −0.061 | 0.998 |
| frozen Phase 6 VAR | 20 | −1.254 | −0.698 | 0.988 |
| frozen Phase 6 VAR | 100 | −0.520 | −0.017 | 0.998 |
| frozen latent MLP | 20 | −1.802 | −1.351 | 0.972 |
| frozen latent MLP | 100 | −7.446 | −7.453 | 0.499 |
| frozen latent kNN | 20 | −1.465 | −1.059 | 0.988 |
| frozen latent kNN | 100 | −1.099 | −0.908 | 0.917 |

Per-session latent H20 skills:

| Session | H20 skill | 95% bootstrap CI |
|---|---:|---:|
| 0 | −1.972 | [−2.048, −1.901] |
| 1 | −4.704 | [−4.900, −4.503] |
| 2 | −0.378 | [−0.471, −0.288] |
| 3 | −0.911 | [−0.951, −0.869] |
| 4 | −0.118 | [−0.135, −0.100] |

Only session 2 at H100 has a positive skill (+0.244, CI [0.186, 0.303]). This is one of ten session-horizon cells and is not a replicated result.

## 4. Interpretation

The latent LDS and the corrected stable VAR both **fail** the autonomous-rollout gate at H20. The stable VAR is slightly less bad than the latent model at H20 but still negative in four of five sessions and at H100 median. Frozen Phase 5/6 baselines are trained on the blocked split, so their comparison is descriptive only, not a fair head-to-head on the new chronological split.

The latent model improves persistence in one session/horizon cell. That is not enough to claim a generative simulator. The dominant failure mode is rapid drift away from the observed marginal distribution; high support q99 fractions do not contradict that because support is a nearest-neighbour envelope, not a forecast-accuracy measure.

## 5. Reproduction

```powershell
python scripts/context_latent_baseline.py
```

Outputs: `model_selection.json`, `latent_dimension_validation.csv`, `stable_var_validation.csv`, `latent_model_selection.csv`, `free_running_latent.csv`, `free_running_latent_per_start.csv`, `trajectories/latent_lds_worm*.npz`.
