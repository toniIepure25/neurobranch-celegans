# Held-out execution plan

Local Windows CPU and pinned Python 3.13.5, NumPy 2.3.1 and SciPy 1.16.0. Use the worktree root and `PYTHONPATH=src`. The runner verifies raw hashes against the freeze manifest and validates protocol/split/seed metadata and runtime versions before proceeding. DEV is the default mode. Use `--validate` in DEV to verify the freeze without generating episodes. No remote resource is needed.

The independent unit is a fresh episode. Each held-out block draws one training stream of 768 episodes and a separate scoring stream of 512 checkpoints. Nested prefixes use 48, 192 and 768 training episodes. The four conditions share streams inside each block. There are 200 independent held-out blocks. DEV runs four blocks with 12, 48 and 96 training episodes and 72 scoring checkpoints per block. Development is spare validation data. It cannot become part of the frozen held-out run. Repeated time steps are never treated as replicates. Scoring has no Monte Carlo outcome or oracle error; block-level uncertainty covers fitted samples and fresh score checkpoints under the frozen prior. Mechanism assignment, initial condition, velocity transitions and training label draws are all represented at the episode level. [Splits](splits_v2.json)

PCG64 uses SeedSequence entropy `20261125` and keys `(namespace, zero-based block, role)`, DEV namespace 11, held-out namespace 29, training role 0, score role 1. No held-out RNG is initialized during freeze. File fingerprints are raw SHA-256 for provenance. The scientific fingerprint hashes canonical UTF-8 JSON of the LF-normalized protocol text and the key-sorted config/split/seed documents. It excludes time and other execution metadata. A local CPU interval of 120 seconds is checked between blocks. [Seeds](seeds_v2.json) · [Freeze manifest](PROTOCOL_V2_FREEZE_MANIFEST.sha256)

The following command is prepared. Its execution requires later explicit user authorization, supplied out of band as a nonempty value of the `NEUROBRANCH_IC1_V2_HELDOUT_AUTHORIZATION` environment variable. The gate is procedural. The value is neither stored nor printed. The runner reserves the fixed new output `research/ic1_v2/HELDOUT_RESULTS_V2.json` in exclusive creation mode. An existing artifact blocks a rerun. If an interruption leaves the reserved artifact incomplete, preserve it and review the failure rather than retrying automatically.

```powershell
Set-Location 'D:/ComputaCenter/NeuroBranch/neurobranch-ic1-v2'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python scripts/ic1_msc_v2.py --mode heldout
```
