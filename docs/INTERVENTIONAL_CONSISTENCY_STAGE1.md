# Stage One: interventional consistency, implemented

**Branch:** `research/interventional-consistency`, from `547e7b1`.
**Protocol:** [`research/interventional_consistency_protocol.md`](../research/interventional_consistency_protocol.md) (`ic1`, frozen; SHA256 in the adjacent `.sha256` file).
**Code:** `src/neurobranch/abstraction/`. **Report:** `reports/interventional_consistency/`.

> Synthetic ground truth only. No claim here concerns *C. elegans*, and no external
> method was reproduced. See `reports/interventional_consistency/CONFORMANCE.md`.

## What exists now

| piece | module | purpose |
|---|---|---|
| information contract | `abstraction/contract.py` + `schemas/information_contract.schema.json` | 12 declared channels; refuses a "representation quality" comparison when access differs |
| estimands | `abstraction/estimands.py` | sup-based behavioural/dynamic commutation error with an occupancy floor and a support guard |
| ground truth | `abstraction/systems.py` | exactly-lumpable chain, ring state space with a witness, sampled rate circuit |
| oracle | `abstraction/reference.py` | exact enumeration over kernels; never calls the simulator |
| sampler | `abstraction/commutation.py` | simulation-based re-estimation and a rollout estimator for the non-enumerable system |
| conformance | `abstraction/conformance.py` | what actually ran, in the `verified`/`simulated`/`unidentifiable`/`falsified`/`not_reproduced` vocabulary |
| pipeline | `abstraction/stage1.py`, CLI `abstraction stage1` | runs every fixture, evaluates the gates, writes the bundle |

## Result of the frozen gates

Run `python -m neurobranch.cli.main abstraction stage1` to reproduce; `--quick` gives
the same verdict with smaller samples. Nine gates, all passing on this machine:
independence of the enumerating reference, agreement between sampling and
enumeration, the positive control, the two negative controls, the history ladder, the
contract refusal, the identifiability witness, determinism, and the frozen history.

Three of them are the substantive content:

- **G3b — history ladder.** `ring/L1` has `ε_D = 0.70`; `ring/L2` has `ε_D = 0.0`
  exactly. One symbol leaves the velocity ambiguous; two symbols contain the
  displacement and fix it. This is the positive control for hypothesis A.
- **G5 — witness.** Under `asym` the error is `0.85` at `L = 1`, `2` and `4`: it does
  not shrink at all, because the operator's effect depends on a context the
  observation never reveals. The correct report is `not_identified`, and the pipeline
  produces exactly that.
- **G1b — two estimators.** Re-estimating the conditional distributions from
  trajectories recovers `0.176` (exact `0.18`), `0.502` (exact `0.50`) and `0.700`
  (exact `0.70`), and lands on the same side of the tolerance every time. The zero of
  the positive control comes back as `0.018`, which is Monte Carlo bias, not a
  disagreement — the protocol compares against the declared noise floor for that
  reason.

## Two findings that changed the design

Both were forced by the fixtures failing, and both are recorded in protocol §14.

1. **A worst-case supremum over all cell members cannot test hypothesis A at all.**
   The product state space of an augmented chain always contains a latent value
   incompatible with the observed window, and that pair supplies the worst total
   variation at every history length. The estimand therefore restricts the supremum to
   members carrying at least `1e-6` of the cell's peak occupancy — and that occupancy
   has to be measured under the **passive** kernel. Measuring it under the intervened
   kernel let an operator delete the context it disagreed with and shrink its own
   error to zero: the benchmark would have certified the abstraction it was supposed
   to catch.
2. **Commutation is a property of the pair (abstraction, operator), not of the
   abstraction.** On the continuous circuit the worst-case error saturates at `1` for
   any hard partition, so the graded quantity is the occupancy-weighted mean. It is
   `0.474` under passive dynamics, `0.396` under `lesion` and `0.257` under `clamp`:
   the very same coarse-graining commutes *better* under two of the interventions than
   under no intervention. Any statement of the form "abstraction X is
   interventionally consistent" is therefore ill-posed without naming the operator
   family, which is one of the design decisions still open with Denis.

## What is deliberately absent

No learned method is evaluated yet. The pipeline runs the oracle, the estimands and
the fixtures; `pca_observation`, `pca_observation_history`, `behaviour_supervised` and
`pca_known_H` are *declared* in the contract set but not yet scored, because scoring
them before the instrument is validated would be exactly the positive-result chasing
the programme forbids. That is the first task of Stage Two, and it is only justified
because G1–G7 pass.

## How to run it

```bash
python -m neurobranch.cli.main abstraction contracts --check   # contracts + refusal of a privileged comparison
python -m neurobranch.cli.main abstraction stage1 --seed 0     # full run, exit 1 on NO-GO
python -m pytest tests/abstraction -q                          # 58 tests
```

Artifacts land in `reports/interventional_consistency/` (`metrics.json`,
`commutation_by_cell.csv`, `SUMMARY.md`, `CONFORMANCE.md`, `manifest.json`,
`manifest.sha256`) and `artifacts/interventional_consistency/ic1/`. Timestamps are
confined to the manifest, so `metrics.json` is byte-identical across reruns.
