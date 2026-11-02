# Baseline reproduction

The interaction audit starts by reproducing the frozen J0/J1/J2/J5 H20 results
on all 175 test landmarks before fitting any new model. It imports the historical
`scripts/jd_evaluate.py` functions rather than reimplementing the mean models,
innovation samplers, PCA, or rollout order.

## Scope

- Kato et al. (2015) `WT_NoStim`, five repeated sessions, 15,662 frames.
- 103 predictors in four sessions and 102 in one session.
- Splits and purge gaps inherited unchanged from the frozen joint-dynamics
  analysis; 175 overlapping test landmarks, stride 10, H20.
- Fixed 10-D PCA fitted only on training frames.
- 100 ensemble members; seed `7000 + session*10000 + landmark`.
- J0 persistence, J1 independent AR (window 200), J2 population VAR with diagonal
  innovations, J5 corrected population VAR with lead-1 block-bootstrap
  innovations (window 500, ridge 1.0, spectral cap 0.995).

## Result

`reports/simulator/kato_interactions/baseline_reproduction.csv` compares every
reproduced metric with `reports/simulator/joint_dynamics/scores_by_landmark.csv`.

| Check | Result |
| --- | --- |
| Rows rerun | 700 (4 models x 175 landmarks) |
| Frozen rows inspected | 1,050 (including J3/J4) |
| Maximum absolute difference, all audited metrics | `8.88e-16` |
| Tolerance | `1e-6` |
| Gate G1 | `PASS` |

The maximum difference is floating-point roundoff. There is no tolerance-driven
reproduction claim.

## J2/J5 equivalence

J2 and J5 share the same conditional mean operator but differ in how innovations
are generated. They are **mean-equivalent, not per-landmark identical**:

- mean signed H20 energy difference (J2 minus J5): `+0.000134`;
- session-block 95% interval: `[-0.0266, +0.0247]`;
- mean absolute H20 energy difference: `0.1258`;
- maximum absolute landmark difference: `0.7495`;
- correlation of landmark scores: `0.9922`;
- mean signed trajectory difference: `-0.00233`.

A single innovation realization can move one landmark score substantially even
when the conditional mean is the same. The audit therefore reports both mean
equivalence and the per-landmark spread; it does not say J2 and J5 are identical
simulators.

## Reproduce

```powershell
cd D:\ComputaCenter\NeuroBranch\neurobranch-celegans
python scripts/kato_interactions_main.py --surrogate-draws 6
python scripts/validate_kato_interactions.py
```

The run writes `reproduction_summary.json` and leaves the frozen
`reports/simulator/joint_dynamics/` artifacts untouched.

## What this does not establish

Exact reproduction of four historical baselines does not validate the new
cross-neuron, mode, state, or perturbation claims. It only establishes that the
new experiment starts from the same numerical object. G7 remains `FAIL` because
the frozen joint calibration and trajectory gates fail.
