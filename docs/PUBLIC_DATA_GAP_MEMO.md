# Public-data gap memo

**Status:** decision memo, 2026-11-11. No new model was trained for this memo.

## 1. Decision

Stop fitting new architectures to the same five worms and the same reused public cohorts. Public
data can support **descriptive external checks** and **response-prediction benchmarks**, but none of
the available cohorts can close the sixth-worm, independent-eight-behaviour, or randomized-
intervention gaps.

## 2. What each public cohort can and cannot support

| Cohort | Verified content | Can support | Cannot support |
|---|---|---|---|
| Kato 2015 | 5 wild-type worms, 15,662 frames | descriptive dynamics; label-parity checks | independent behaviour; held-out animal claim; intervention |
| Baskoylu & Flavell 2026 | 27 animals, 43,200 frames, 1,162 reversal events, CC0 | independent binary reversal external check; kinematic descriptives | eight-state labels; exact 11-neuron Tier-3 readout (0/27 animals); sixth BunDLe worm |
| Atanas / Worm Wide Web | 30 recordings | neural replay; readout descriptives | exact 11-neuron readout (0/30); full clamped validation beyond the 3 retained recordings |
| Randi 2023 | 110 animals, 3,371 valid stimulation events | response prediction; target-holdout checks | behaviour; matched behavioural intervention; causal `do()` |
| Kramer 2026 | turning/behaviour cohort | D/V descriptives | shared BunDLe embedding; matched intervention |

## 3. Verified coverage limits

Readout panel for Tier 3:

```text
F = mean(AVBL, AVBR, RIBL, RIBR, RID)
R = mean(AVAL, AVAR, RIML, RIMR, AVEL, AVER)
z = F − R
```

- Baskoylu: **0/27** animals contain all 11 neurons; **8/27** contain the six reverse neurons.
- Atanas: **0/30** recordings contain all 11; **3/30** contain the six reverse neurons.
- Kato: all five animals were used in the original analysis; the labels are derived from
  AVA/SMD/RIB, so they are not independent video labels.

Therefore the exact Tier-3 readout cannot be evaluated on an independent public cohort. A reduced
reverse-only panel is possible in eight Baskoylu animals and three Atanas recordings, but it is a
different endpoint and must not be presented as Tier-3 validation.

## 4. Existing negative results that should not be rerun

- External binary reversal audit: readout mean AUC 0.848, but persistence mean AUC 0.939; the model
  lag-0 mean AUC is 0.633. The model does not beat persistence.
- Mirrored prospective audit: mean AUC 0.388. The clamped model does not forecast reversal.
- `β = 0` is not consistently worse, so the external check does not establish a necessary
  gap-junction contribution.
- Baskoylu incremental neural signal: balanced accuracy 0.6113 with behaviour+neural versus 0.6100
  with behaviour only; log loss 0.4056 versus 0.4087. This is not a robust incremental signal.
- Order-2 temporal holdout: ΔNLL 0.0183, 95% CI [0.0055, 0.0342], positive in 5/5 worms but
  individually resolved in only 2/5. Predictive only; no external replication.

Rerunning these with another seed or another architecture would not change the design limitation.

## 5. Allowed next public-data work

Only if Denis explicitly requests it:

1. a **pre-registered** reduced reverse-panel sensitivity analysis on Baskoylu, with animal-level
   CIs and a stated non-equivalence to Tier 3;
2. a **pre-registered** response-prediction benchmark on Randi, with target-holdout splits and no
   behavioural claim;
3. a descriptive comparison of Atanas recordings that pass the full intrinsic-neuron gate.

None of these is a substitute for G1/G2/G3. They are secondary analyses, not gap closures.

## 6. Conclusion

The next decisive result requires new data: a sixth exact embedding, independent video annotations,
or a randomized intervention. The project should spend its remaining effort on those artifacts and
on the contracts that make them usable, not on another model fit.
