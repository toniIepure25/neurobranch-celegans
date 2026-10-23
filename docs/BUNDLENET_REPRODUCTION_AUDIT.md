# BunDLe-Net reproduction audit (Phase 2)

**Status:** artifact verification + partial causal-filter reimplementation.
**Artifacts:** `artifacts/reproduction_bundlenet_official.json`,
`artifacts/reproduction_bundlenet_lite.json`
**Code:** `src/neurobranch/representation/bundlenet.py`,
`src/neurobranch/reproduction/ncmcm.py`

> This is **not** a full reproduction of BunDLe-Net. Upstream code was not
> executed end-to-end. The official 3-D embeddings distributed with NC-MCM were
> evaluated as artifacts, and a PyTorch reimplementation was trained under a
> causal filter and blocked split. Those are two different things and their
> numbers must not be conflated.

## 1. Official artifacts are transductive

The NC-MCM repository distributes files named
`bundlenet_consistent_embedding_worm_<i>.npz`. They contain a 3-D embedding for
every frame. The upstream embedding was fitted on **all frames**, including the
frames later used as the held-out test block.

Consequences:

- fitting a kNN classifier on train labels and scoring test labels does **not**
  remove the leakage: the test embeddings were shaped by test frames;
- the high kNN accuracy is an in-sample geometry result;
- silhouette and trajectory geometry are likewise fitted on all frames.

The result is reported only as an **upper-bound artifact check**, not as
generalization.

### Official artifact numbers (all five worms)

| Worm | kNN accuracy | kNN macro F1 | 3-D silhouette (fine) | PCA-3 silhouette |
|---:|---:|---:|---:|---:|
| 0 | 0.9841 | 0.9758 | 0.4858 | −0.0082 |
| 1 | 0.9809 | 0.9756 | 0.5467 | 0.0120 |
| 2 | 0.9739 | 0.8787 | 0.5359 | 0.1040 |
| 3 | 0.9789 | 0.9541 | 0.5651 | 0.0634 |
| 4 | 0.9487 | 0.8954 | 0.4465 | 0.0299 |

Range: kNN **0.9487–0.9841**, silhouette **0.4465–0.5651**. These are consistent
with the published artifact quality and are **not** evidence that a causally
trained model recovers the same geometry.

## 2. BunDLeLite: what was actually trained

A PyTorch reimplementation on worm 0 with:

- flattened 15-frame window encoder 50-30-25-10-3;
- residual latent transition `T_Y`;
- linear 8-state behaviour predictor;
- `gamma = 0.9` weighted Markov consistency + behaviour cross-entropy;
- Gaussian noise `sigma = 0.05`;
- blocked split (not upstream 7-fold random KFold);
- **causal** one-pass 4th-order low-pass at the upstream normalized cutoff
  (instead of upstream zero-phase `filtfilt`, which is acausal);
- no PCA initialization / best-of-5;
- 120 epochs (not 2000);
- Keras `Normalization` left as identity, matching upstream's unadapted layer.

### BunDLeLite result (worm 0)

| Metric | Value |
|---|---:|
| kNN accuracy | 0.4067 |
| kNN macro F1 | 0.3039 |
| 3-D silhouette (fine) | −0.0642 |
| Trustworthiness | 0.7570 |
| Best epoch | 16 |
| Best validation loss | 0.17783 |
| Train / test windows | 1,853 / 612 |

The causal reimplementation does **not** reproduce the official geometry. Its
silhouette is slightly negative and its kNN accuracy is less than half the
leaked artifact value.

An earlier acausal run reached kNN 0.6236 / macro F1 0.4740 / silhouette 0.1452.
That run is retained only as a **leakage counterexample**: the improvement came
from filtering test frames into earlier inputs. It must not be cited as a
reproduction result.

## 3. Why this is a partial reproduction, not a reproduction

Missing or changed relative to upstream:

| Upstream | This audit |
|---|---|
| TensorFlow/Keras | PyTorch |
| 7-fold random KFold | blocked contiguous split |
| zero-phase `filtfilt` | causal one-pass filter |
| PCA initialization, best-of-5 | none |
| 2000 epochs | 120 epochs |
| full training pipeline / schedules | reduced |
| exact published preprocessing | data adapter approximation |

No claim is made that the published BunDLe-Net accuracy was reproduced. The
audit establishes only that (a) the distributed artifact is leaky, and (b) a
causal-filter version does not recover the same geometry on this pipeline.

## 4. Implication for NeuroBranch

The Phase 1 claim “latent geometry does not clearly beat persistence” stands.
The official BunDLe numbers cannot be used as a counterexample because they are
transductive. Any future claim of beating Phase 1 must use a causal filter and a
strictly held-out encoder, or explicitly label the result as transductive.

## 5. Reproduction

```bash
# official-artifact evaluation (no retraining)
python -c "import json; from neurobranch.representation.bundlenet import official_vs_reimplementation as f; print(json.dumps([f(w) for w in range(5)], indent=2))"

# causal reimplementation
neurobranch reproduce bundlenet --worm 0 --epochs 120
```

The upstream repository is not vendored. See
[`REPRODUCING.md`](REPRODUCING.md) for the external artifact paths.
