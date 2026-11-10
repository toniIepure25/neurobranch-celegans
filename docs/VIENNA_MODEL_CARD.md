# Vienna Neuroinformatics Studio — model card

## Models in the product

| Tier | Model | Fitted on | Output | Claim level |
|---|---|---|---|---|
| 1 | BunDLe-Net 3-D embedding + quadratic ridge dynamics + logistic behaviour read-out | 5 published worms; leave-one-worm-out | 3-D latent trajectory, 8 behaviour probabilities | Descriptive latent-manifold forecast |
| 1 | Manifold walk (kNN transition sampler) | same 5 worms | latent trajectory sampled from observed successors | Descriptive; does not extrapolate |
| 1 | Switching linear dynamics + behaviour Markov chain | same 5 worms | latent trajectory + behaviour sequence | Descriptive; behaviour-conditioned |
| 2 | NC-MCM cluster transition model | precomputed 5-worm NC-MCM artifacts | state/behaviour distributions under a forced state | Observational; `do()` not identified |
| 3 | Clamped premotor rate ODE | published A/d + recovered W/names; Atanas recording | 15 intrinsic traces, behaviour readout | Implementation-faithful replay/sandbox; not a digital twin |
| 3 | 205-node free-run premotor ODE | same matrices, no clamping | 205-node activity | Exploratory numerical sandbox |

## Intended use

- Explore and compare the group's published modelling objects on a held-out worm.
- Teach the difference between a latent-manifold forecast, an observational state model, and a
  biophysical replay.
- Generate hypotheses and pre-register experiments; export traces and reports for discussion.

## Out-of-scope use

- Clinical or BCI decision-making.
- Claims of whole-brain mechanism, causal identification, or a validated digital twin.
- Extrapolation to a sixth worm or to a new species without re-fitting and held-out validation.
- Treating the 205-node free-run or a signal pulse as an experimentally validated intervention.

## Evaluation

- Tier 1: leave-one-worm-out; 25-frame open-loop horizon; 16 seeds; latent MSE, CRPS, marginal
  50/90% ensemble coverage, behaviour accuracy/NLL/Brier, transition JS; persistence and marginal
  baselines.
- Tier 2: Markov-order Monte Carlo test, stationarity block bootstrap, held-out behaviour
  transport, state–behaviour mutual information, K sensitivity.
- Tier 3: published parameter values, 84/15/69 canonical counts, β=0 and zero-signal controls,
  intrinsic residual and behaviour readout. Fresh Julia 1.11.2 runs on all three validation
  recordings match the port at mean trace r=0.9999998 and max |Δ|=0.0133; behaviour-label parity is
  99.98%. Behaviour agreement with the recorded neural readout is 60.2% canonical, 58.3% mean, so
  this is **not** a pointwise biological-fidelity claim.

## Known limitations

- Only five worms; worm-level splits are correct but population claims are limited.
- Ensemble intervals under-cover nominal 90% in all three Tier-1 models.
- NC-MCM strict first-order Markov and stationarity are rejected by the current audit.
- Premotor `W` and neuron names are recovered, not part of the public JLD2 export. The Python port
  matches the authors' Julia code on all three validation recordings; this validates the
  implementation, not the model's biology or causality. The behaviour target is the paper's
  neural readout proxy; it tracks independent reversal events at AUC 0.848 / 75.9%, but the
  clamped model only reaches AUC 0.633 / 55.9% on the same annotation.
- Clamped recordings other than `2023-01-23-15` have shorter measured cores (85 and 86 here); their
  counts are reported as measured and are not forced to 84/15/69.

## Provenance

BunDLe-Net DOI `10.1101/2023.08.08.551978` (preprint); NC-MCM DOI
`10.1371/journal.pcbi.1012674`; Celegans_premotor DOI `10.1371/journal.pcbi.1013818`.
Third-party artifacts are restored by `python scripts/fetch_vienna_artifacts.py`; they are not
committed because their licenses differ.
