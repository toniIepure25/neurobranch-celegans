# Kato final audit: decomposition and model comparability

## 1. Arithmetic

The reported decomposition is an ordered sequence:

```text
B4m  ->  B5-10  ->  B3
```

Using mean H20 PCA-10 energy:

| Term | Energy | Contrast | Value |
| --- | ---: | --- | ---: |
| B4m own AR (500) | 3.239667 | — | — |
| B5-10 own + ten factors | 2.964588 | common factor increment, B4m − B5 | +0.275079 |
| B3 population VAR | 2.700150 | residual increment, B5 − B3 | +0.264438 |
| total | — | B4m − B3 | +0.539517 |

The arithmetic is exact:

```text
0.2750793961657263 + 0.2644379926683560 = 0.5395173888340823
```

The reported shares are:

```text
common    = 0.275079 / 0.539517 = 0.509862  (51.0%)
residual  = 0.264438 / 0.539517 = 0.490138  (49.0%)
```

These percentages are **successive increments in a chosen model order**, not
unique variance components. The total is defined relative to B4m, a deliberately
weak own-history baseline.

## 2. Order dependence

Changing only the baseline changes the shares while leaving the B5−B3 residual
increment unchanged:

| Baseline | Total advantage | Common increment | Residual increment | Common share | Residual share |
| --- | ---: | ---: | ---: | ---: | ---: |
| B4m (reported) | +0.539517 | +0.275079 | +0.264438 | 51.0% | 49.0% |
| B1m own AR (200) | +0.345764 | +0.081326 | +0.264438 | 23.5% | 76.5% |
| B0 persistence | +0.956808 | +0.692370 | +0.264438 | 72.4% | 27.6% |

Therefore the “51/49” split is not a stable property of the system. It is a
property of the ordered B4m→B5→B3 comparison, the ten-factor representation,
ridge estimation and the choice of B4m as baseline. It does not identify
physiological common input, direct coupling, or neuron-specific interactions.

## 3. Model definitions and comparability

| Dimension | B3 population | B5-10 common + own | B4m own (500) | B1m own (200) |
| --- | --- | --- | --- | --- |
| Available history | full current population, 500-frame causal window | target own current + ten shared factor scores, 500-frame window | target own current, 500-frame window | target own current, 200-frame window |
| Parameters per neuron | `n+1` outputs, total `n(n+1)` | 12 per neuron, total `12n+110` | 2 | 2 |
| Parameters, `n=103` | 10,712 | 1,346 | 206 | 206 |
| Parameters, `n=127` | 16,256 | 1,634 | 254 | 254 |
| B3/B5 parameter ratio | 7.96–9.95× across sessions | — | — | — |
| Penalty | ridge 1.0, standardized features | ridge 1.0, standardized features | ridge 1.0 | ridge 1.0 |
| Intercept | unpenalized | unpenalized | unpenalized | unpenalized |
| Innovation | common J5, same per landmark | common J5, same per landmark | common J5 | common J5 |
| Initialization | same current frame | same current frame | same current frame | same current frame |
| Evaluation | same 175 H20 landmarks, same PCA-10 | same | same | same |
| Forecast | 100-step rollout, H20 scored | factor VAR + diagonal own rollout | own AR rollout | own AR rollout |

### Matched

- Same recordings, landmarks and future targets.
- Same J5 innovation draws for the paired comparison.
- Same 500-frame causal window for B3, B4m and B5-10.
- Same ridge strength on standardized features.
- Same unpenalized intercept and forecast initialization.
- Same ten-dimensional PCA evaluation space.

### Not matched

- **Capacity.** B3 has 7.96–9.95× more parameters than B5-10. B5 is a restricted
  diagonal-plus-factor model, not a refitted B3.
- **Information set.** B5 has the target's own current value and all ten factor
  scores, but it cannot represent arbitrary other-neuron residual coordinates.
- **Baseline strength.** B4m is weaker than B1m; the reported 51/49 split uses
  B4m, while the stronger own-history baseline gives a different split.
- **Representation.** The ten-factor space is fixed at `k=10`; factor selection
  is not repeated on the test block.
- **Parameter uncertainty.** The comparison is a fixed-model contrast, not a
  nested likelihood-ratio or variance-component estimate.

## 4. Conclusion

The arithmetic is correct and the paired evaluation is matched on data,
innovations, window, penalty, initialization and metric. The models are **not**
matched on capacity or information set, and the percentages are **not unique**.

The defensible statement is:

> In the ordered B4m→B5-10→B3 comparison, common factors account for +0.275
> energy units and the remaining B5−B3 contrast is +0.264; this is a descriptive
> decomposition, not a causal or unique variance partition.

It is not defensible to say “51% of the population advantage is biological common
input” or “49% is direct neuron-to-neuron coupling.”
