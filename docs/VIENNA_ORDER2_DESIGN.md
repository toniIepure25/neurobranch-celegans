# NC-MCM order-2 follow-up — design and result record

Status: implemented; predictive result only. This does not identify a causal mechanism and does not
rescue the `do()` claim.

## Question

The existing audit rejects a strict first-order Markov chain. If extra history is real, an explicit
second-order chain should predict held-out frames better than a first-order chain.

## Design

- Unit: one worm; state sequence at the audit's selected `K` (default 6).
- Split: first half of each recording for transition counts, second half for scoring.
- First test context may include pre-split history, as in an online forecast.
- Additive smoothing `alpha=0.5`; identical smoothing for both orders.
- Metric: mean negative log-likelihood on the common held-out frames; positive
  `ΔNLL = NLL(order 1) − NLL(order 2)` favours order 2.
- Uncertainty: moving-block bootstrap over per-frame NLL differences, block 50, 300 resamples
  per worm; aggregate over worms with a worm-level bootstrap.
- No test frame is used to choose `K`, the split, the smoothing, or the model.

## Decision criterion

- `supports_order2` only if the aggregate 95% CI excludes zero and the direction is positive.
- Report the per-worm count, including worms that do not resolve individually.
- Do not call this causal, mechanistic, or externally validated.

## Result

- Aggregate mean `ΔNLL = 0.0183`, 95% CI `[0.0055, 0.0342]`.
- Positive point estimate in **5/5** worms; individually resolved in **2/5**.
- Per-worm deltas: `0.0001, 0.0245, 0.0475, 0.0056, 0.0136`.

Interpretation: a small, consistent predictive gain from one extra frame of history. The strict
first-order assumption is still rejected, stationarity still fails, and `do(state)` remains a
model-based counterfactual. External-worm replication is required before any stronger claim.

## Implementation

`src/neurobranch/bundlenet_sim/causal_audit.py`:

- `_context_index`, `_markov_nll` — order-1/order-2 temporal-holdout NLL.
- `_block_bootstrap_ci` — moving-block uncertainty.
- `markov_order_forecast` — per-worm comparison.
- `markov_order_forecast_all` — aggregate verdict.
- `audit_all` — writes `tests.markov_order_forecast`.

Regenerate:

```powershell
python -m neurobranch.bundlenet_sim.causal_audit `
  --out reports/vienna_workbench/tier2_assumption_audit.json
```
