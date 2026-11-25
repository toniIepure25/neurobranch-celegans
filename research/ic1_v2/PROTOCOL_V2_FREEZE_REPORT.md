# IC1-v2 minimal experiment freeze report

**READY FOR HELD-OUT AUTHORIZATION**

Freeze preflight completed on 25 November 2026 at about 13:16 Europe/Bucharest, on detached HEAD `65793350c9d3eb2f6afda6f76c271b912d474d39` in `neurobranch-ic1-v2`. The 12 pre-existing IC1-v2 files were found untracked and unstaged, and their recorded preflight SHA-256 digests were verified before writing the new manifest. All 12 match byte-for-byte and remain in their original paths. The main, historical IC, and H100 worktrees retained their initial Git status and HEADs. No competing Python or Git process referring to the isolated worktree was found at preflight. The old reports and their hashes remain historical evidence. No tracked edits or staged files were added.

| Frozen identity | SHA-256 |
|---|---|
| Scientific fingerprint, normalized protocol and canonical config/split/seed JSON | `e7a412b002a3e860263806df1436205bf2fa91859dc543e9301ffec48f545303` |
| Raw provenance manifest, 34 referenced paths | `3ce12ce2d68b2c33820a7fa25de35634c25437bcd421ec355fac5d05e2218b66` |
| Runtime plus thread settings, canonical JSON | `4dbb3f43e783508004f2034dd139f2fa8af64f05c17d2645429bd952d220b508` |

The runtime identity used Windows, Python 3.13.5, NumPy 2.3.1, SciPy 1.16.0 and one OpenBLAS/OpenMP thread. The manifest includes runtime dependency declarations, the runner, its oracle, test code, config, split and seed files, and all existing IC1-v2 evidence. The scientific fingerprint preserves substantive text and config changes while normalizing LF and CRLF and JSON key order. The timestamp stays outside that fingerprint. No pre-existing file changed, so its before and after hash is the same manifest entry.

## Gates

| Gate | Outcome |
|---|---|
| A–H. Historical calibration and numeric instrument | The 13-case, 19,968-evaluation replay passed and reproduced the prior acceptance JSON at SHA-256 `c2b67a20abc3d7876b7854a2b1c1960dc8015d3166d9f88cc9a8168055a91f68`. Full scoped Ruff passed. The abstraction suite passed **97 tests**, including 11 new freeze tests. The historical CLI smoke test's output directory was redirected in memory to the isolated pytest scratch directory. The historical four-state oracle/sampler/repair values remained 1/0/1. Positive rare support, intervention-only bridge diameter 1 and persistent-step diameter 0 all passed. |
| I–K. Loss and exact analytical risks | Balanced Bayes averages over the equiprobable mechanisms and their histories. Minimax maximizes over the two fixed mechanisms after taking each within-mechanism expectation. Exact recurrence and rational certificates give 17/40 without sensor, 0 with full history and sensor, and 7/20 after compression with sensor. The oracle agrees on all 12 rows with the independently implemented IC1-v2 kernel oracle. It remains unchanged if a learner prediction is deliberately corrupted. |
| L–P. Information and sampling | Supervised single-intervention prediction uses identical train/score episodes across conditions, only the assigned information resolution and sensor access, and one label per training episode. Test changes to hidden mechanism and velocity fields leave permitted inputs unchanged. Sensor masking and matching are verified. The learned comparator is a tabular conditional-response predictor. TV uses exact response probabilities. Seed namespaces, block indices and role keys are disjoint, with fresh IID episodes and nested training prefixes within each block. |
| Q–T. Execution boundary | The unauthorized held-out branch refuses before manifest verification, RNG creation and generation. The private run path also checks authorization. Development uses only namespace 11. The authorized held-out branch was never invoked and no held-out outcomes were generated or inspected. IC2 was not executed or integrated. Historical worktrees were preserved. Kubernetes was not used. |

The base compressed optimum is also 17/40, so the base information gap is zero under this loss. Its conditional mean distribution has TV risk 19/40, an objective gap of 1/20. The sensor-arm information gap is 7/20. The unrestricted categorical table has zero approximation gap. The fitted excess over its cell oracle combines objective mismatch with finite-sample/smoothing deviation, which is reported separately against the population conditional-mean rule. Exact belief references are privileged by their true model, prior and response laws. Their inference inputs remain restricted to the stated cells.

## Development evidence and reproducibility

Default DEV and explicit `--mode dev` each ran four independent blocks. Each block has 96 training episodes at the largest prefix and 72 scoring checkpoints. Each run fitted 48 tables across the three nested sample sizes and four conditions. Score outcomes were not generated. DEV suppressed the scientific agreement decision. The two runs took about 1.20 and 1.17 seconds and produced identical temporary output bytes, SHA-256 `0e7e7137f773a62c5a27640227f511ba8a9153cda11bbe6fca4ff6adf1574f32`. The independent instrument replay took about 34 seconds. No code or config was tuned using a held-out result.

In the worktree root, set the environment shown in the [execution plan](HELDOUT_EXECUTION_PLAN.md). The scoped validation commands were:

```powershell
python scripts/ic1_msc_v2.py --validate
python scripts/ic1_msc_v2.py --output $nbFirst
python scripts/ic1_msc_v2.py --mode dev --output $nbSecond
python scripts/ic1_v2_acceptance.py $out
python -m pytest -q -p no:cacheprovider tests/abstraction/test_ic1_msc_freeze.py tests/abstraction/test_ic1_v2.py
python -m ruff check src/neurobranch/abstraction/ic1_msc_oracle.py scripts/ic1_msc_v2.py tests/abstraction/test_ic1_msc_freeze.py src/neurobranch/abstraction/ic1_v2_contract.py src/neurobranch/abstraction/ic1_v2_oracle.py src/neurobranch/abstraction/ic1_v2_sampling.py scripts/ic1_v2_acceptance.py tests/abstraction/test_ic1_v2.py
```

The DEV and calibration output variables pointed to new unique files in the system temporary directory. For the final full abstraction suite, use this in-memory redirect to preserve the old default metrics artifacts:

```python
from pathlib import Path
import pytest
from neurobranch.abstraction import stage1

stage1.ARTIFACT_DIR = Path.cwd() / ".pytest_tmp" / "ic1_v1_cli_scratch"
raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "tests/abstraction"]))
```

## Ready to request execution

The paired 95% marginal t interval uses 200 independent train/evaluation blocks. Per-block training prefixes are 48, 192 and 768; the primary contrast uses 768 and 512 paired scoring checkpoints. SeedSequence entropy is 20261125 with separate DEV and held-out namespaces and distinct train and score roles. The held-out command is `python scripts/ic1_msc_v2.py --mode heldout` after the frozen local environment and a later explicit authorization. Its guard is procedural. The first authorized run will verify end-to-end held-out execution. Scientific claims stay within the two specified mechanisms and constructed sensor; the DEV interval and the conservative IC1-v2 coverage grid have separate roles. [Frozen protocol](MINIMAL_DECISIVE_EXPERIMENT_PROTOCOL_V2_FROZEN.md) · [Freeze manifest](PROTOCOL_V2_FREEZE_MANIFEST.sha256) · [Execution plan](HELDOUT_EXECUTION_PLAN.md)
