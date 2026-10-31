# Simulator results

> **Bottom line.** A functioning free-running simulator exists and is
> unit-tested. On the frozen gates it shows **no dynamical advantage** over
> null baselines, **no branching advantage**, and **no specific named-neuron
> perturbation effect**. Gate 6 (biological validation) is **not evaluable** on
> Kato 2015. This is a deliberate negative result, not a failed attempt to hide
> one.

## 1. What was run

| Item | Value |
|---|---|
| Dataset | Kato et al. 2015, `WT_NoStim.mat`, five wild-type worms |
| Frames | 15,662 total |
| Predictors | 103–127 per worm; 31–47 named; eight behaviour-label neurons excluded |
| Split | blocked, five blocks, test block 2, validation block 4, purge 12 |
| Models | persistence, VAR, MLP, MLP-residual, GRU, latent MLP, latent kNN |
| Checkpoints | 105 (7 kinds × 5 worms × 3 seeds) |
| Selected kind | `mlp_residual` by validation one-step R² |
| Environment | Python 3.13.5, CPU PyTorch 2.7.1 |

The run is an **interpolation** evaluation, not a forward forecast and not a
causal experiment. All perturbation outputs are model-dependent virtual edits.

## 2. Gates

From `reports/simulator/gates.json`, schema-validated by
`schemas/simulator_gates.schema.json`:

| Gate | Question | Status |
|---|---|---|
| G1 | Does the simulator produce valid free-running rollouts from `x0`? | **pass** |
| G2 | Does it beat persistence / constant-state nulls on held-out trajectories? | **fail (0/5 worms)** |
| G3 | Does it reproduce conditional reversal branching better than nulls? | **fail (0/5)** |
| G4 | Can a named neuron be perturbed through an explicit interface? | **pass (interface only)** |
| G5 | Are perturbation effects robust and specific versus controls? | **fail** |
| G6 | Is there biological validation against an intervention dataset? | **not_evaluable** |

Decision: `functional_simulator_no_dynamical_or_branching_advantage`.

Passing G4 does **not** mean a biological perturbation works. It means the
software can represent one and report a model-dependent effect.

## 3. G1: the simulator works

- All five worms produce finite rollouts for the selected model and seed 17.
- `x(0)` is preserved exactly.
- No future observation is read: a test replaces future frames with `NaN` and
  the rollout is unchanged.
- Same seed reproduces; different seeds can differ when noise is positive.
- Shapes, provenance, support diagnostics and manifest fields are checked.

G1 is a software/contract gate, not a scientific validity gate.

## 4. G2: no free-running advantage

The selected model is `mlp_residual`, chosen on validation one-step R²
(0.7134). On the held-out test block:

| Kind | Free-running R² | MSE − constant baseline | 5-step MSE | OOD fraction |
|---|---:|---:|---:|---:|
| `var` | **0.5325** | **−0.0491** | **0.3594** | **0.0076** |
| `persistence` | 0.4368 | 0.0000 | 0.4572 | 0.1600 |
| `gru` | 0.3726 | +0.0925 | 0.4557 | 0.0456 |
| `mlp` | 0.3078 | +0.1639 | 0.4868 | 0.0694 |
| `latent_knn` | 0.2283 | +0.2368 | 0.5782 | 0.3032 |
| `latent_mlp` | 0.1420 | +0.2956 | 0.4807 | 0.2854 |
| `mlp_residual` (selected) | **−0.5098** | **+0.8398** | 0.5145 | **0.7903** |

The constant-state baseline is the initial state repeated for the whole horizon.
It is the correct free-running null; one-step persistence is a different
quantity and is reported in `one_step_metrics.csv`.

Gate 2 required mean ΔR² ≥ 0.02, a seed-17 block-bootstrap CI above zero, and
5-step MSE no worse than the constant baseline in ≥4/5 worms. No worm passed.
The selected model's mean ΔR² across three seeds is about +0.0079, but the
per-worm values are mixed (−0.0041, −0.0068, +0.0016, +0.0011, +0.0478) and its
5-step MSE is worse in every worm (worm 4: 0.6835 vs 0.6778).

The descriptive winner is **VAR**, not the selected model. VAR beats the
constant baseline overall and has very low OOD, but it was not selected by the
one-step rule and does not become a causal simulator by winning this table.

## 5. G3: no branching advantage

Evaluated on 93 non-overlapping held-out reversal states, horizon 20, 100
rollouts per state:

| Worm | Observed entropy | Simulated entropy | TV sim | TV persistence | TV marginal |
|---|---:|---:|---:|---:|---:|
| 0 | 0.482 | 0.107 | 0.713 | 0.555 | 0.328 |
| 1 | 0.584 | 0.000 | 0.657 | 0.660 | 0.285 |
| 2 | 0.680 | 0.084 | 0.551 | 0.443 | 0.170 |
| 3 | 0.837 | 0.011 | 0.541 | 0.546 | 0.194 |
| 4 | 0.609 | 0.004 | 0.662 | 0.586 | 0.251 |

- The marginal null is closer than the simulator in all five worms.
- The simulator collapses outcomes to near-zero entropy.
- Observed block-permutation nulls show an entropy **deficit** (worms 0, 1, 4:
  p = 0.01, observed below null), not an excess. No worm shows significant
  branching excess.

See [BRANCHING_SIMULATION.md](BRANCHING_SIMULATION.md).

## 6. G4/G5: named-neuron perturbations

Six pre-registered candidates: AVBL, AVBR, AVEL, AVER, RIVL, RIVR. Main sweep:
600 perturbed conditions, 300 initial-state probes, 1,800 dose, 900 timing,
1,800 model-seed, 900 random-control rows. Ten starts per worm and 10 rollouts
give 100 paired rollouts per target.

- Top mean `|ΔTV|`: AVEL inhibition 0.146, RIVL activation 0.102, AVBR
  activation 0.058, AVBL activation 0.048, RIVR activation 0.046.
- Random-control 95th percentile of `|ΔTV|`: **0.500**. No candidate exceeds it.
- No candidate agrees in sign in ≥4/5 worms. The best is 3/5 (RIVL activation).
- Perturbed OOD fraction is **0.98–0.99**; mean trajectory displacement is
  7.5–8.0 training SD. The model is extrapolating almost everywhere.
- Initial-state probes move trajectories by about 2.5 training SD but change
  outcome distributions very little on average (`ΔTV` 0.002–0.020).
- Dose–response is not monotonic in the expected direction with a specific
  effect for any candidate.

The AIBL/AIBR pilot is disclosed as a protocol deviation and is not used for
Gate 5.

See [NEURONAL_PERTURBATION_METHODS.md](NEURONAL_PERTURBATION_METHODS.md).

## 7. Latent geometry

PCA 3-D explains 0.386–0.531 of training variance. The selected neuron-space
model's PCA path is too short (ratio 0.186 vs observed) and too straight
(curvature difference −1.18 rad). A train-only BunDLe encoder is unstable across
seeds (CKA 0.102 and 0.284; Procrustes residuals 1.02 and 2.03), so it is not
used as a latent coordinate system. BunDLe has no published inverse decoder;
named-neuron perturbations are applied in neuron space and projected only for
display.

See [LATENT_SIMULATION_VALIDATION.md](LATENT_SIMULATION_VALIDATION.md).

## 8. Negative findings, explicitly

1. A high one-step validation R² does not imply stable free-running dynamics.
   The selected MLP-residual model has the best validation one-step R² and the
   worst free-running test error.
2. No model robustly beats the constant-state baseline across all five worms;
   VAR is the best descriptive exception.
3. The simulator does not reproduce reversal branching better than a
   marginal-outcome null.
4. Observed reversal data show an entropy deficit relative to a block null, not
   the entropy excess that a branching narrative would want.
5. No named-neuron virtual perturbation is specific versus random controls or
   robust across worms and model seeds.
6. Nearly all perturbed states are outside the training support.
7. BunDLe latent geometry is not stable enough to anchor a latent intervention
   claim.
8. There is no empirical stimulation-response model, so Gate 6 cannot be
   evaluated and category C is refused by code.

## 9. Figures

The PNGs live at `figures/simulator/` in the repository (not inside `docs/`, so
they are not copied into the MkDocs site):

```text
figures/simulator/trajectory_pca.png
figures/simulator/model_comparison.png
figures/simulator/branching.png
figures/simulator/perturbation.png
figures/simulator/latent.png
```

They are regenerated by `python scripts/simulator_figures.py`.

## 10. Exact reproduction

From the repository root on Windows PowerShell:

```powershell
python scripts/simulator_fit.py
python scripts/simulator_evaluate.py
python scripts/simulator_latent.py
python scripts/simulator_branching.py
python scripts/simulator_perturb.py
python scripts/simulator_figures.py
python scripts/simulator_gates.py
python scripts/validate_reports.py --reports reports/simulator --strict
python scripts/simulator_manifest.py
```

The latent script trains seven BunDLe encoders (five worms plus two stability
seeds) and takes several minutes on CPU. The perturbation sweep is the other
long step. Existing Phase 2 checkpoints are reused; `--fit-missing` only fills
absent ones.

## 11. What would change the conclusion

A positive result requires an independent cohort, a calibrated outcome model, a
support-preserving perturbation, a random-control comparison, and ideally a
matched stimulation dataset. None is available in this checkout.
