# Stage-One protocol: interventional consistency of neural abstractions

**Version:** `ic1`
**Frozen:** 2026-11-24, before any Stage-One run. The three refinements listed in
§15 were made during implementation and before the first reported number; they are
recorded there rather than silently absorbed.
**Branch:** `research/interventional-consistency`, base commit `547e7b1`.
**Status:** pre-registered analysis plan. No number produced under this protocol is
a finding until the gates in §9 pass.

## 1. Question

Under which conditions does a macrostate representation `C_t = τ(H_t^{(L)})`, learned
from partial observations of a known micro-level system, preserve the system's
dynamics and the effects of interventions — and when is that impossible from the
information the protocol permits?

Scope: synthetic, mechanism-known systems with declared observation models. This
protocol makes no statement about *C. elegans*.

## 2. Objects

Micro state `X_t ∈ X`; intervention `u ∈ I`; intervention kernels
`P_u(X_{t+1} | X_t)`; observation model `Y_t ~ O(· | X_{0:t})` (may include calcium
filtering); behaviour readout `B_t ~ Q(· | X_t)`; finite history
`H_t^{(L)} = (Y_{t-L+1}, …, Y_t)`; macrostate `C_t = τ(H_t^{(L)})`; macro-level
intervention map `ω: I_X → I_C`.

The representation is **not** assumed memoryless. `L = 1` is one point in the grid,
not the definition.

## 3. Estimands

Primary, per macro cell `c` and operator `u`, with `D` total variation for discrete
states and `W_1` on a declared grid for continuous states:

1. **Behavioural commutation error** `ε_B(c,u)`: supremum of `D` between
   `P(B_{t+1} | do(u), H_t=h)` and `P(B_{t+1} | do(u), H_t=h')` over `h, h'` with
   `τ(h) = τ(h') = c`.
2. **Dynamic commutation error** `ε_D(c,u)`: the same supremum for
   `P(τ(H_{t+1}) | do(u), H_t=·)`.

**Occupancy restriction (part of the estimand).** The supremum runs over pairs of
cell members that both carry at least `δ = 1e-6` of the cell's peak occupancy, where
occupancy is measured under the **passive** kernel. Without the restriction the
supremum is always attained by a pair of members that no trajectory can reach in the
same cell — the product state space contains latent values incompatible with the
observed window — and then no amount of history can ever reduce the error, which
would make hypothesis A untestable by construction. Measuring occupancy in the
reference regime rather than in the intervened one is equally load-bearing: an
operator that makes one context unreachable would otherwise be allowed to shrink its
own error by deleting the alternative it disagrees with.
3. **Worst-case macro inconsistency** `max_c ε(c,u)` — reported always, because the
   mean hides aliasing.
4. **Macro–micro commutation** `Δ_commute(u) = D[τ# P_X^u(X_{t+1} | X_t),
   P_C^{ω(u)}(C_{t+1} | C_t)]`, defined only when `ω(u)` is identified; otherwise the
   run reports `not_identified` with the reason.

Secondary, never substituted for a primary: one-step and multi-step prediction
error; behaviour-prediction accuracy; Markov diagnostic (lag-2 test statistic and
p-value); latent reconstruction error; and the occupancy-weighted **mean** pairwise
total variation, which is the only graded quantity for a hard partition of a
continuous state (the supremum there saturates at one).

Aggregation: per operator, over cells with support. The independent unit for
uncertainty is the **cell × seed block**; cells within a seed are not independent.

## 4. Information contract

Every method declares, machine-readably, whether it receives each of: current
observations, observation history, future observations, behaviour labels, neuron
identities, missingness masks, measurement operator `H`, true latent state,
intervention type, intervention target, intervention magnitude, test-animal
adaptation.

Comparison rule: two methods may be compared as representations only if their
contracts are identical, with `history_length` the single pre-specified exception.
Any other difference makes the comparison *conditional on information access* and it
must be labelled as such. Methods with `true_latent_state = true` are privileged
upper bounds and are excluded from any ranking.

## 5. Systems and controls

Three families, each with three pre-declared variants. Fixtures are code, not data.

| Family | Positive control | Behaviour-only negative | Intervention-only negative |
|---|---|---|---|
| `exactly_lumpable` (discrete Markov) | exactly lumpable partition | behaviour preserved, dynamics not | observationally compatible, distinguished only by one operator |
| `rate_circuit` (4 populations + a hidden modulator, self-contained) | operators compatible with the visible populations | behaviour readout preserved under a coarse partition | hidden modulator changes the effect of `u` |
| `state_space_calcium` (latent modulator + calcium observation) | history-sufficient (`L*` finite) | behaviour preserved, `ε_D > 0` | no permitted history separates the mechanisms |

The third variant in `state_space_calcium` is the **identifiability witness**: it is
constructed so that a stated macro-level equivalence holds by construction, and the
protocol must report `not_identified` rather than a small error. Equivalence is
*designed*, not statistically proven, and the report says so.

## 6. History grid

`L ∈ {1, 2, 4}`, declared before running. The augmented state space grows as
`6 × contexts × 3**L`, so `L = 8` alone would need a 78k × 78k kernel: longer windows
are out of scope for Stage One rather than silently approximated. For each family the
report gives `ε(L)` and the smallest `L` whose error is at or below the tolerance
(`0.05`).

## 7. Estimation and uncertainty

- Cells: macro cells with at least `MIN_CELL_COUNT = 30` sampled pairs. Cells below
  are counted in `n_unsupported` and excluded; they are never imputed or merged.
- Pairs: up to `N_PAIRS = 200` within-preimage pairs per cell, drawn with a
  documented scheme; paired starts use **common random numbers** so that the pair
  difference is not inflated by independent noise.
- Uncertainty: percentile bootstrap, `B = 2000`, resampling cells within an operator
  and seeds within a cell. Nominal 95%.
- Support coverage (fraction of target mass in supported cells) is a first-class
  reported metric, not a footnote.
- Determinism: one `seed = 0`; artifacts must be byte-identical across reruns on the
  same environment apart from timestamps confined to the manifest.

## 8. Metrics reported per run

`ε_B`, `ε_D`, worst case, `Δ_commute` (or `not_identified`), support coverage,
`n_unsupported`, prediction error, behaviour accuracy, Markov statistic and p-value,
reconstruction error, interval width, and the contract label of every method.

## 9. Technical gates

**GO** requires all of:

| Gate | Criterion |
|---|---|
| G1a oracle independence | `exact_commutation` completes with `DiscreteFixture.simulate` replaced by a raising stub |
| G1b estimator agreement | the sampling estimator reproduces the enumerator on every agreement fixture within its declared noise floor, and lands on the same side of the tolerance |
| G2 positive control | exactly-lumpable family gives `ε_B = ε_D = 0` up to numerical tolerance |
| G3 negative controls | each declared negative gives `ε > 0.05` for the intended estimand and `ε ≤ 0.05` for the one it is supposed to preserve |
| G3b history ladder | `ring` gives `ε(L=1) > 0.05` and `ε(L=2) = 0` exactly |
| G4 contract enforcement | a comparison across differing contracts is refused with a non-zero exit |
| G5 identifiability witness | under `asym` the error stays above `0.05` at every `L` and does not shrink below half its maximum |
| G6 reproducibility | rerun is byte-identical apart from timestamps |
| G7 frozen history | hashes of the historical phase manifests match the recorded baseline |

**NO-GO** (methodological failure, must be fixed before scaling): any of G1–G7
failing; oracle and simulator sharing an implementation path; a negative control not
detected; a result that depends on the decoder while claiming to be about the
representation.

A **scientifically negative** outcome — history does not help, commutation does not
predict transfer, NC-MCM and PCA are indistinguishable on the tested families — is
**not** a technical NO-GO. It is a result, and it is reportable.

## 10. Forbidden before G1–G7 pass

Any statement that an abstraction is "interventionally consistent" in general; any
ranking across contracts; any biological, cognitive or causal claim; any use of the
words *digital twin*; any hyperparameter search to make a control separate.

## 11. Deviations

Recorded in `research/interventional_consistency_deviations.md` with date, reason and
affected gate **before** the affected run. An undocumented deviation invalidates the
run. No post hoc threshold changes.

## 12. Artifacts

`artifacts/interventional_consistency/ic1/{metrics.json, commutation_by_cell.csv,
manifest.json, manifest.sha256}`; `reports/interventional_consistency/{SUMMARY.md,
CONFORMANCE.md, metrics.json}`. The manifest records the git revision, library
versions and the SHA256 of every input and output.

## 13. Base-commit inventory (why Stage One is self-contained)

The base commit `547e7b1` tracks evidence for phases 1–15, but the P2–P8.5 report
directories and the entire `src/neurobranch/mechanistic/` package are **untracked**
there. Consequences, frozen here:

1. Stage One must not import `neurobranch.mechanistic.*`; the rate circuit is
   implemented inside `abstraction/systems.py` as a *new* fixture family, not as a
   reproduction of the P2 circuit.
2. The frozen-history baseline in §G7 covers **tracked** manifests only. The absence
   of P2–P8.5 from version control is itself reported as a reproducibility finding,
   not repaired by this branch.
3. Conformance may read vendored third-party artifacts (`references/external/`,
   gitignored) because they are inputs to the comparison, not project evidence.

## 14. Changes made before freezing, and why

Three design decisions were settled during implementation, before any number was
reported. They are listed here because each one changes what the estimand *means*.

1. **Occupancy restriction.** The first implementation took the supremum over all
   members of a cell. The history ladder then failed for a trivial reason: the
   product state space always contains a latent value incompatible with the observed
   window, so the worst pair never improved with `L`. Restricting the supremum to
   reachable members (and measuring reachability in the passive regime) is what makes
   hypothesis A testable at all.
2. **`ring_state_space` replaced an earlier emission model.** A noisy two-symbol
   emission of a persistent latent was tried first. It cannot produce a positive
   result for history, because a window with balanced symbols leaves the posterior
   balanced at every `L`; the supremum therefore stays at its `L = 1` value forever.
   That is a legitimate *negative* system, but it cannot also serve as the positive
   control, so the ring (where displacement identifies velocity exactly) took its
   place.
3. **`L = 8` and `L = 16` dropped from the exact grid.** They would require kernels of
   `78k²` and `2.7e7²` entries. Claiming them without a sampling estimator would be
   fabrication; approximating them silently would be worse.

## 15. What would falsify the benchmark itself

- If the exactly-lumpable positive control cannot reach zero error, the estimand is
  mis-specified.
- If every negative control separates equally well for trivial reasons (e.g. the
  partition is degenerate), the benchmark measures partitioning, not abstraction.
- If commutation error is perfectly predicted by prediction error on all families,
  the new estimand adds nothing and Stage Two should say so.
