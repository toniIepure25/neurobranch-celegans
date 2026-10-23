# Behavioural generalization (Phase 2)

**Status:** executed on the five Kato recordings, blocked and strict-forward
protocols.
**Artifacts:** `reports/phase2a_scientific_audit/behavior/`
**Code:** `src/neurobranch/evaluation/behavior_benchmark.py`

> The behaviour labels were derived from AVA, SMD and RIB activity in the
> original dataset. Those eight cells are excluded from predictors, but the
> task remains partly circular. This is a decoding benchmark, not a causal
> claim that neural activity generates behaviour.

## 1. Question

Does neural activity add information **beyond the current behavioural state**?
Phase 1 compared classifier accuracy to label persistence, which can hide the
fact that most frames do not change state. This audit uses:

- current-state one-hot logistic regression;
- a state-only Markov transition model;
- neural-only PCA + logistic regression;
- current state + neural activity;
- metrics on the **transition subset** (`next state != current state`), not just
  overall accuracy.

Features are PCA(20) of the neural predictors, fitted on training frames only.
The classifier is multinomial logistic regression fitted on training frames.
The blocked test block and the strict-forward test block are the same as Phase 1.

## 2. Overall results

Mean over five worms. `log_loss` and `brier` are probabilistic; lower is better.

### Blocked (interpolation) protocol

| Model | Accuracy | Balanced accuracy | Macro F1 | Log loss | Brier | Transition accuracy |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 0.9713 | 0.9583 | 0.7191 | 0.5942 | 0.0573 | 0.0000 |
| Markov state | **0.9713** | **0.9583** | **0.7191** | 0.1334 | 0.0556 | 0.0000 |
| logistic state | **0.9713** | **0.9583** | **0.7191** | 0.1365 | 0.0557 | 0.0000 |
| state + neural | 0.9606 | 0.9436 | 0.7078 | 0.1478 | 0.0683 | 0.1789 |
| neural only | 0.8180 | 0.7746 | 0.5715 | 0.4767 | 0.2636 | 0.2402 |

### Strict-forward protocol

| Model | Accuracy | Balanced accuracy | Macro F1 | Log loss | Brier | Transition accuracy |
|---|---:|---:|---:|---:|---:|---:|
| persistence | **0.9820** | **0.9695** | **0.7714** | 0.3720 | 0.0359 | 0.0000 |
| Markov state | 0.9579 | 0.8736 | 0.6686 | **0.1342** | **0.0536** | 0.0455 |
| logistic state | 0.9574 | 0.8733 | 0.6679 | 0.6329 | 0.0688 | 0.0227 |
| state + neural | 0.9119 | 0.7461 | 0.5378 | 0.8511 | 0.1399 | 0.3011 |
| neural only | 0.7227 | 0.5469 | 0.3961 | 1.3901 | 0.4251 | 0.3693 |

The persistence baseline has the highest accuracy because the forward test block
is dominated by a few states (worm 1 has **zero** transitions in its forward test
block). Its log loss is bad because it emits a hard one-hot distribution; the
Markov model is the honest state-only probabilistic baseline.

## 3. Incremental effect of neural activity

Paired per-worm difference `state + neural − logistic state`:

| Protocol | Metric | Mean Δ | SD | Worms improved |
|---|---|---:|---:|---:|
| blocked | balanced accuracy | −0.0147 | 0.0090 | 0/5 |
| blocked | macro F1 | −0.0113 | 0.0113 | 0/5 |
| blocked | log loss | +0.0114 | 0.0257 | 2/5 (lower is better) |
| blocked | Brier | +0.0126 | 0.0148 | 0/5 |
| blocked | transition accuracy | +0.1789 | 0.0892 | 5/5 |
| forward | balanced accuracy | −0.1272 | 0.1456 | 0/5 |
| forward | macro F1 | −0.1301 | 0.1382 | 0/5 |
| forward | log loss | +0.2182 | 0.3077 | 0/5 |
| forward | Brier | +0.0711 | 0.0581 | 0/5 |
| forward | transition accuracy | +0.2784 | 0.1512 | 4/5 (1 undefined) |

Neural activity **does** help identify that a transition will occur: the model
predicts the correct next state on 17.9% (blocked) and 27.8% (forward) of the
rare transition frames, where the state-only model is at 0% by construction. But
this comes at a cost: every overall calibration and balanced-accuracy metric
gets worse. The model buys transition sensitivity by giving up persistence
accuracy.

This is not evidence of incremental predictive value for behaviour. A model that
is worse on every proper scoring rule and every balanced metric cannot be called
an improvement.

## 4. Transition prevalence

| Protocol | Worm 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| blocked | 3.2% | 2.7% | 2.8% | 1.7% | 4.0% |
| forward | 2.5% | 0.0% | 0.5% | 2.3% | 3.7% |

With ~18 transition frames per worm in the blocked test set, transition accuracy
is extremely noisy. The forward protocol has as few as zero transitions, so no
transition metric is defined for worm 1.

## 5. Conclusion

**RQ3 answer:** no robust incremental neural prediction of behaviour beyond the
current state. Neural activity contains a weak transition signal, but it does not
improve overall behaviour prediction or calibration relative to a state-only
Markov model, in either blocked or strict-forward evaluation.

## 6. Limitations

- Labels are derived from the same neurons used in the recording; despite
  excluding AVA/SMD/RIB, the task is not independent of the neural data.
- The forward split collapses to near-single-state behaviour for some worms.
- The transition subset is tiny; per-worm transition accuracy has wide binomial
  uncertainty that is not resolved by five worms.
- No external independent behaviour dataset (e.g. Atanas/Kim) was acquired and
  validated in this phase. That remains the correct next test.
