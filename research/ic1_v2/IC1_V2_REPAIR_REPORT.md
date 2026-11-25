# IC1-v2 repair report

**READY FOR PROTOCOL FREEZE**

The prespecified finite-synthetic instrument gates passed. The applicable scope is a declared passive start design on labelled states, a fixed operator sequence, and a fixed categorical terminal response. The MSc protocol is ready for review and freeze. Its holdout scoring remains pending separate authorization. IC2 and official reference implementations were outside this run.

## Phase A replay and root cause

Replayed from historical IC1 HEAD `65793350c9d3eb2f6afda6f76c271b912d474d39` in a new detached worktree. The earlier dossier's 21 entries matched the manifest. The historical 58 tests and nine gates reproduced in the CRLF checkout. The four-state example gave oracle **1**, v1 trajectory estimate **0**, and the independent passive-start pilot **1**. With seed 139, 2,000 passive starts yielded endpoint counts [0,1032] and [968,0] for the two positive states. The historical estimator used 40 chains, 50 retained steps, burn-in 20 and seed 0. It visited only the other cell, 2,000 times. The original oracle deliberately evaluated the passive stationary design, with its historically specified threshold.

The historical [sampler](../../src/neurobranch/abstraction/commutation.py) burns in and counts transitions under the **queried operator**, then infers support and weights from those same paths. The [reference](../../src/neurobranch/abstraction/reference.py) uses passive stationary weights for the conditional rows. In the counterexample the operator eventually moves all mass to a single state. That removes precisely the starts responsible for diameter 1. Both the supremum's domain and any occupancy weights can change systematically. More trajectories under the operator cannot recover missing passive rows. The defect applies whenever the relevant start distributions differ. The original IC1 estimated one-step rows, so its multi-step implications are prospective. The new multi-step fixtures test the start-only restriction directly. [Prior dossier](../../../.research/neurobranch-foundations-v1/IC_RESEARCH_AUDIT.md)

The exact ring conclusions stand. L1 diameter 0.70, L2 diameter zero, and historical asymmetric L1/L2/L4 diameters 0.85 reproduced. For the fixed two-context pair, the passive laws agree for any allowed history and the first-probe minimax radius is 0.425. The old technical gate count remains historical evidence. The result requiring correction is generalized sampler–oracle equivalence. V2's weighted expectation is a new quantity and should not be substituted into the old mean-pairwise table.

## Implemented instrument

- **Support diameter** maximizes within-cell terminal-response TV over every state with positive exact integer mass under the passive checkpoint law. It stratifies independent trajectories across these starts.
- **Occupancy-weighted error** averages the same row-level TV over two independent starts conditional on their common passive cell, including equal-start pairs, and weights cells by passive mass. It independently samples start pairs and conditional response trajectories.
- The oracle propagates conditional laws algebraically. The sampling path uses the simulator and empirical categorical counts. A test replaces the simulator to make it give the wrong result while the exact oracle still gives the right result.
- Zero mass excludes a state only as a passive starting point. In the two-step fixture, the first action enters a state outside passive support, and a second action sends it to the only endpoint with response 1. The diameter is 1. Repeating the first action for both steps yields diameter zero. The response alphabet remains fixed in both cases.
- The rare passive state of mass 1e−8 determines diameter 1. The weighted estimate frequently has no such trajectory, and the output names the missing positive start. The historical trimming sensitivity is explicit and reports the excluded mass. A structurally impossible state that shares a cell label never enters the sample plan.
- Constant start abstraction still leaves diameter 1 when the fixed response differs. Changing the target itself to a constant gives diameter zero. The fixture makes that target change explicit.

[Full spec](IC1_V2_ESTIMAND_SPEC.md) · [All 13 fixture outcomes](IC1_V2_ACCEPTANCE_RESULTS.json) · [Bias and coverage](IC1_V2_MONTE_CARLO_CALIBRATION.md)

## Verification and workspace integrity

Commands were run in the detached worktree with `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, and one OpenBLAS and OpenMP thread. The unmodified historical suite ran before implementation. The final combined suite passed 86 tests, 28 of them new. Ruff passed for all five added Python files. Exact commands and sequence:

```powershell
git -C D:/ComputaCenter/NeuroBranch/neurobranch-interventional -c core.longpaths=true worktree add --detach D:/ComputaCenter/NeuroBranch/neurobranch-ic1-v2 65793350c9d3eb2f6afda6f76c271b912d474d39
python -m pytest -q -p no:cacheprovider tests/abstraction
python -c "from pathlib import Path; from neurobranch.abstraction.stage1 import run_stage1, summary_line; print(summary_line(run_stage1(seed=0, out_dir=Path.cwd()/'.pytest_tmp'/'ic1_v1_replay')))"
python -m ruff check src/neurobranch/abstraction/ic1_v2_contract.py src/neurobranch/abstraction/ic1_v2_oracle.py src/neurobranch/abstraction/ic1_v2_sampling.py scripts/ic1_v2_acceptance.py tests/abstraction/test_ic1_v2.py
python -m pytest -q -p no:cacheprovider tests/abstraction
python scripts/ic1_v2_acceptance.py research/ic1_v2/IC1_V2_ACCEPTANCE_RESULTS.json
$nbReplay = Join-Path $env:TEMP ('nb-ic1-v2-replay-' + [guid]::NewGuid().ToString('N') + '.json')
python scripts/ic1_v2_acceptance.py $nbReplay
python -m pytest -q -p no:cacheprovider tests/abstraction/test_ic1_v2.py
```

The historical CLI smoke test redirects its reports but writes a seed-5 metrics file under the default artifacts path in the detached copy. The generated file was retained in temporary storage and that copied path restored from the checkout-filtered HEAD object to SHA-256 `42e2e33d8544aa19d3b7a4942b06bb5d70c8d66153be56c10441de2527599081`. The registered original IC1 worktree and its pre-existing dirty metrics were preserved. No tracked files remain modified in the detached copy. The historical dossier remains its own frozen snapshot. No staged files, commits, registry changes, remotes or Kubernetes activity were created by the repair.

## What remains before scoring

The updated MSc draft fixes the response, mechanism pair, passive sensor and evaluation contract. The strongest fair comparator for this finite single-probe task is the learned tabular action-conditional predictive state. The exact belief-state decision rule is privileged. Under the declared model, the full-history oracle risk drops from 0.425 to zero with the perfect sensor, and discarding the first position leaves a sensor-arm oracle risk of 0.35. These are deductions from the design. The finite-data contrasts still require the proposed independent blocks and score evaluator. [MSc draft](MINIMAL_DECISIVE_EXPERIMENT_PROTOCOL_V2_DRAFT.md) · [Prior-art delta](MINIMAL_EXPERIMENT_PRIOR_ART_DELTA.md)

The single next authorization is approval to execute the first held-out evaluation of the minimal synthetic MSc experiment after freezing its protocol. Model evaluation requires the frozen seed and split manifest, the score evaluator and independent evaluation blocks. IC2 learned representation scoring additionally requires its own matched contracts, evaluator integration and explicit evaluation authorization.
