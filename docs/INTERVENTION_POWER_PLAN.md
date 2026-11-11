# Intervention power and analysis plan

**Status:** design only. No intervention data exist. This plan does not assert that the proposed
sample sizes are achievable in the lab.

## 1. Estimand

For a randomized target-versus-sham experiment, the primary estimand is the **animal-level average
treatment effect** on a pre-registered binary behavioural outcome:

```text
Δ = P(outcome = 1 | target) − P(outcome = 1 | sham)
```

Trials within an animal are repeated measurements, not independent observations. The analysis
therefore uses an animal-level summary or a mixed-effects model with a random intercept per animal.
Frames are never treated as independent.

## 2. Design defaults

| Parameter | Screening | Confirmation |
|---|---|---|
| animals per arm | 6 | 12 |
| arms | target, sham, random-neuron (where ethical) | target, sham |
| trials per animal | 30 | 30 |
| randomization | within animal, blocked | within animal, blocked |
| blinding | scorer blinded to arm | scorer blinded to arm |
| primary outcome | `P(turn within 4 s | reversal)` | same, frozen |
| secondary | bout duration, D/V turn, latency, calcium response | same |
| ICC | 0.2 (assumption) | 0.2 (assumption) |
| alpha | 0.05 two-sided | 0.05 two-sided |
| target power | 0.80 | 0.90 |

The ICC is an assumption, not a measurement from these data. The power report must show
sensitivity to ICC ∈ {0.05, 0.1, 0.2, 0.4}.

## 3. Current simulated design (no data)

`python scripts/power_intervention.py --reps 300` produced
`reports/vienna_workbench/intervention_power.json` at the default design ICC of 0.2:

| Design | MDE (absolute risk difference) |
|---|---|
| 6 animals/arm, 30 trials | 0.30 |
| 12 animals/arm, 30 trials | 0.20 |

At 6 animals/arm, a 0.20 absolute effect has simulated power 0.55 at ICC 0.2. These numbers are
design assumptions, not a promise about the lab. They say that the screening design is only
adequate for a large effect; a 0.20 effect needs the confirmatory design or more animals.

## 4. Power simulation

`scripts/power_intervention.py` simulates a cluster-randomized binary outcome:

1. draw an animal-level random effect `u_a ~ Normal(0, σ²)`, with `σ² = ICC/(1−ICC)`;
2. draw trial outcomes with arm-specific probabilities;
3. analyse with a cluster-level test or mixed-effects logistic regression;
4. repeat with a fixed seed;
5. report power and the minimum detectable effect (MDE) for the default design.

The script must write:

```json
{
  "format": "vienna-intervention-power-v1",
  "seed": 20261111,
  "assumptions": {"design_icc": 0.2, "trials_per_animal": 30, "alpha": 0.05},
  "designs": [{"animals_per_arm": 6, "power": 0.0, "mde": 0.0}],
  "sensitivity": [{"icc": 0.05, "power": 0.0}]
}
```

The shape above is illustrative; the actual values are computed by the script and stored in
`reports/vienna_workbench/intervention_power.json`. No result is fabricated in docs.

## 5. Analysis

- Primary: mixed-effects logistic regression with random intercept per animal, arm as fixed effect,
  pre-specified baseline covariate if available.
- Fallback: animal-level proportions + cluster bootstrap; randomization test as a check.
- Report: effect size, 95% CI, number of animals, number of trials, exclusions, ICC estimate.
- Replication: a confirmatory cohort is required before a causal claim. A single experiment is
  hypothesis-generating even if significant.
- Negative result: report the CI and the MDE; do not reinterpret a null as evidence of no effect.

## 6. Positive control

The assay needs a positive control with a known behavioural direction. Without it, a null target
result is uninterpretable. The positive control must be specified by the lab before data
collection and analysed separately.

## 7. Stop rules

- Stop if the positive control fails.
- Stop if arm balance or clock alignment fails validation.
- Stop if the target cannot be identified reliably in the recording.
- Do not add animals after seeing the effect; the confirmatory sample is fixed in advance.
- Do not switch the primary endpoint after unblinding.

## 8. What this cannot establish

A successful experiment establishes a causal effect of **that intervention, in that assay, in that
genotype/preparation**, after replication. It does not validate the whole-brain model, all eight
behaviours, or a digital twin.
