# Literature Review: Whole-Brain Dynamics, Latent Manifold Learning, and Causal Abstraction in *C. elegans*

**Status:** literature review for NeuroBranch (C. elegans population dynamics, branching analysis, virtual perturbation).
**Compiled:** 2026-10-23.
**Companion:** [`docs/RESEARCH_FOUNDATIONS.md`](RESEARCH_FOUNDATIONS.md) — formal problem statement, assumptions, RQ1–RQ7.

> **Post-review implementation note (2026-10-23).** After this review was
> written, NeuroBranch verified the distributed BunDLe-Net embeddings and NC-MCM
> artifacts and ported the NC-MCM Markov test to Python. The review's source
> claims remain *not independently reproduced*: artifact inspection and a test
> port are not a retraining of either method. See
> [`RESULTS.md`](RESULTS.md) and [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md)
> for what was actually run.

---

## 1. Scope and method

**Decision informed.** NeuroBranch fits predictive models to whole-brain calcium imaging, analyses where population trajectories diverge ("branching"), and runs virtual perturbations in the fitted model. The project is a candidate collaboration with the Neuroinformatics group at the University of Vienna (M. Grosse-Wentrup), which produced both BunDLe-Net and NC-MCM. The question is therefore narrow: which published methods are baselines, which are competitors, and which claims must not be repeated without new evidence?

**Review question.** Given only observational whole-brain calcium imaging of immobilised *C. elegans* (Kato et al. 2015), what is established about low-dimensional neural dynamics and behaviour transitions, what can latent-variable and switching-state models add, and what would be required before trajectory branching or virtual perturbations may be described as causal or as bifurcations?

**Procedure.** `web_search` was unavailable (no search API key). Sources were resolved directly through Crossref (`api.crossref.org/works/<doi>`), Europe PMC REST (`resultType=core`, which returns abstracts), the arXiv API, publisher pages, and the three local repositories. Every DOI below was resolved and its title, venue, year, volume and pages checked against resolver output. Two textbooks are cited for standard definitions and checked via OpenLibrary ISBN; they are not counted among the 20 primary works. This is a **targeted, not systematic, review**: no PRISMA screening, no citation chasing, no meta-analysis.

**Evidence tags.** **[E]** established by the cited source (abstract read; full text for BunDLe-Net and NC-MCM); not independently reproduced. **[A]** artifact-verified by reading files in this checkout — no training run, no recomputed metric. **[H]** my inference, offered as a hypothesis. **[P]** proposed NeuroBranch contribution.

**No result here was reproduced by NeuroBranch.** Reading a repository is not reproduction; inspecting an `.mat` file is not reanalysis.

---

## 2. Source table

Venue/year as returned by the resolver. "Data" is what the source actually analysed. Limitations are the authors' own where stated, plus structural limits evident from the method.

| # | Source (venue, year) | Contribution | Data | Evaluation reported | Main limitations | Relevance |
|---|---|---|---|---|---|---|
| 1 | [Kumar, Gilra, Gonzalez-Soto, Meunier & Grosse-Wentrup, *bioRxiv* 2023](https://doi.org/10.1101/2023.08.08.551978) | BunDLe-Net: window encoder τ, residual latent transition T_Y, 8-class readout, weighted commutativity loss | Kato WT_NoStim, 5 worms | Dynamic prediction error vs. autoregressive baseline; behaviour decoding; vs. PCA, t-SNE, AE, ArAe, CEBRA | Preprint; one dataset; 3-D chosen for display; "Markov by construction" is architectural; no perturbation | Primary baseline; source of window = 15, γ = 0.9 |
| 2 | [Grosse-Wentrup, Kumar, Meunier & Zimmer, *PLoS Comput Biol* 2024](https://doi.org/10.1371/journal.pcbi.1012674) | NC-MCM: cognitive states as causally consistent abstractions; 28 pairwise logistic regressions → k-means → Markovianity test | Kato WT_NoStim, 5 worms (107–131 neurons reported) | 10-fold CV decoding; Markovianity p over k = 1…20, 100 restarts, 1 000 surrogates | Observational only → behavioural *observational* consistency; p maximised over restarts; labels from motor-neuron activity | Discrete abstraction + Markov test; complement to branching analysis |
| 3 | [Morrison & Young, *PLoS Comput Biol* 2025](https://doi.org/10.1371/journal.pcbi.1013818) | Connectome-based premotor model: 15 core neurons, >100 drivers, cubic intrinsic dynamics, ReLU synapses, diffusive gap junctions | Atanas 2023 imaging, White connectome, voltage clamp | Reproduction of switching, synchrony, higher-order statistics, heat-shock effects, roaming/dwelling | Core group hand-selected; parameters partly tuned; no causal validation | Strongest mechanistic null model; nearest thing to a bifurcation account |
| 4 | [Kato et al., *Cell* 2015](https://doi.org/10.1016/j.cell.2015.09.034) | Cyclical global brain state whose segments map to motor commands; defines the 8 WT states | 5 immobilised worms, no stimulus, 18 min each | Single-trial mapping to motor commands; state annotation | Immobilised; labels inferred from AVA/SMD/RIB activity; no perturbation in this subset | The primary dataset |
| 5 | [Atanas et al., *Cell* 2023](https://doi.org/10.1016/j.cell.2023.07.035) | Brain-wide activity and motor programs in freely moving worms; per-neuron encoding models; connectome-class atlas | >30 recordings via WormWideWeb | Cross-validated encoding models; identified-neuron atlas | Encoding, not intervention; partial identification | Natural external validation set with real behaviour |
| 6 | [Leifer et al., *Nat Methods* 2011](https://doi.org/10.1038/nmeth.1554) | Tracking optogenetics in freely moving worms (ChR2/NpHR, ~50 fps, ~30 µm); motor, egg-laying, mechanosensory circuits | Freely moving transgenics | Acute behavioural effects of stimulation/inhibition | Few cells; light/heat artefacts; state-dependent efficacy | The empirical standard for "intervention" |
| 7 | [Linderman et al., arXiv 2016](https://arxiv.org/abs/1610.08466) | Recurrent switching linear dynamical systems: discrete modes, logistic transitions, per-mode linear-Gaussian dynamics | Neural spiking, synthetic | Mode recovery, decoding | Fixed mode count; linear per mode; no behaviour supervision | Canonical "discrete branch + continuous dynamics" model |
| 8 | [Krishnan, Shalit & Sontag, arXiv 2016](https://arxiv.org/abs/1609.09869) | Deep Markov model: nonlinear Gaussian latent state space, structured variational inference | Music, medical time series | ELBO fit; held-out prediction | No discrete modes; identifiability unestablished | Explicit latent-confound baseline |
| 9 | [Chen, Rubanova, Bettencourt & Duvenaud, *NeurIPS* 2018](https://arxiv.org/abs/1806.07366) | Neural ODEs: continuous-depth dynamics, adjoint gradients, latent ODE-RNN | Time series, density estimation | Accuracy; memory/speed trade-offs | Deterministic; no process noise; solver cost | Continuous-time formulation for ~3 Hz sampling |
| 10 | [Tzen & Raginsky, *ICLR* 2019](https://arxiv.org/abs/1905.09883) | Neural SDEs as latent Gaussian models in the diffusion limit, with variational bound | Synthetic, time series | Bound quality; prediction | Diffusion-limit theory; costly; weak identifiability | Formal object for separating noise from instability |
| 11 | [Pandarinath et al., *Nat Methods* 2018](https://doi.org/10.1038/s41592-018-0109-9) | LFADS: sequential VAE, GRU generator, controller inputs, Poisson likelihood | Motor-cortex spiking | Single-trial rate reconstruction; decoding | Needs trials/alignment; deterministic generator | Single-trial latent baseline; Kato data have no trials |
| 12 | [Kaplan et al., *Neuron* 2020](https://doi.org/10.1016/j.neuron.2019.10.037) | Behavioural hierarchy across three timescales implemented by nested dynamics | Brain-wide imaging in behaving worms | Cross-timescale coupling | Circuit-specific; not a general estimator | Branch points should be organised across timescales |
| 13 | [Cook et al., *Nature* 2019](https://doi.org/10.1038/s41586-019-1352-7) | Complete whole-animal connectomes of both sexes | Electron microscopy | Anatomical completeness | Anatomy only, no dynamics | Prior for connectome-constrained models and ablation targets |
| 14 | [Lappalainen et al., *Nature* 2024](https://doi.org/10.1038/s41586-024-07939-3) | Connectome-constrained deep network of the fly optic lobe (64 cell types), optimised neuron/synapse parameters | Fly connectome + 26 studies | Activity predictions across 26 studies | Task must be specified; fly-specific; needs sparse connectivity | Shows connectome + task constraint can predict activity |
| 15 | [Costa et al., *PNAS* 2024](https://doi.org/10.1073/pnas.2318805121) | Behaviour alone admits a high-fidelity Markov model over posture microstates; coarse-graining via slow eigenvectors | High-resolution video | Prediction across scales; new behaviours; foraging trade-off | No neurons; Markov order fixed | Crucial null: behaviour can look Markovian without a neural latent |
| 16 | [Brennan & Proekt, *eLife* 2019](https://doi.org/10.7554/eLife.46814) | Loops in phase space; two macroscopic variables predict future motor commands up to 30 s ahead, across individuals | Kato recordings | Future motor commands; dwell times; neuron activation | Descriptive; heuristic loop extraction; no intervention | Closest precedent for transition prediction on this dataset |
| 17 | [Wilting & Priesemann, *Nat Commun* 2018](https://doi.org/10.1038/s41467-018-04725-4) | Subsampling severely biases branching-ratio estimates; derives an invariant estimator | Cortex; epidemics | Bias correction; reanalysis | Depends on avalanche generative model | 109–135 of 302 neurons is severe subsampling |
| 18 | [Beggs & Plenz, *J Neurosci* 2003](https://doi.org/10.1523/JNEUROSCI.23-35-11167.2003) | Neuronal avalanches, power laws, critical branching | Cortical slices | Power-law fits, avalanche shape | Slice preparation; later statistical debate | Origin of the "branching" vocabulary |
| 19 | [Schölkopf et al., *Proc IEEE* 2021](https://doi.org/10.1109/JPROC.2021.3058954) | Causal representation learning: independent mechanisms; identifiability conditions | Conceptual | — | Identifiability needs interventions or multiple environments | A latent is a coordinate system, not automatically a cause |
| 20 | [Runge et al., *Nat Rev Earth Environ* 2023](https://doi.org/10.1038/s43017-023-00431-y) | Causal discovery for time series: Causal Markov, sufficiency, stationarity, no instantaneous effects; PCMCI+ | Climate/geoscience | Benchmarked discovery | Assumptions rarely testable; latent confounders dominate | Assumption checklist before causal language |

**Textbooks** (definitions only, not counted above): Pearl, *Causality*, 2nd ed., CUP 2009 ([ISBN 9780521895606](https://openlibrary.org/isbn/9780521895606)); Peters, Janzing & Schölkopf, *Elements of Causal Inference*, MIT Press 2017 ([ISBN 9780262037310](https://openlibrary.org/isbn/9780262037310)).

---

## 3. BunDLe-Net: what the artifact implements

**Idea.** Learn a low-dimensional embedding that preserves only the dynamical and behavioural information relevant to a behavioural context [E]. Mechanism: encoding the next observation and evolving the current latent estimate must land in the same place, and behaviour must be decodable from that landing point.

**Architecture** (`references/external/BunDLe-Net/functions.py`) [A]:

- **τ**: `Flatten` → `Dense(50, relu)` → `Dense(30, relu)` → `Dense(25, relu)` → `Dense(10, relu)` → `Dense(latent_dim, linear)` → `Normalization` → `GaussianNoise(0.05)`. With ~100 neurons and 15 frames the input is ≈1 500-D; `latent_dim = 3`.
- **T_Y**: `Dense(latent_dim, linear)` → `Normalization`, used **residually**: `Y_{t+1} = Y_t + T_Y(Y_t)`. A first-order deterministic map — not an RNN, not a neural ODE.
- **Predictor**: one `Dense(8, linear)` over the eight WT states.
- **Two arms**: the upper arm encodes the next window and decodes behaviour; the lower arm encodes the current window, applies T_Y, and is compared with the upper encoding.
- **Loss**: `γ·MSE(Y_{t+1}^{upper}, Y_{t+1}^{lower}) + (1−γ)·SparseCategoricalCrossentropy(B_{t+1}, logits)`, `γ = 0.9`. Dynamics dominate 9:1.

**Preprocessing** [A]. `Database` reads `deltaFOverF_bc`, transposes to `(time, neurons)`, and optionally drops eight labelling neurons (`AVAR, AVAL, SMDVR, SMDVL, SMDDR, SMDDL, RIBR, RIBL`). `preprocess_data` applies a 4th-order Butterworth filter forward and backward with `f_h = 0.05`, `f_l = 1e-10`. Since the helper multiplies by `sampling_freq/2` before passing the value to `signal.butter(..., fs=sampling_freq)`, the effective cut-off is `0.05 × 2.9/2 ≈ 0.0725 Hz` — matching the preprint's stated 0.07 Hz [E/A]. The docstring calls these "angular frequencies"; the arithmetic is Hz-equivalent, and easy to misconfigure. `prep_data(win=15)` builds paired 15-frame windows offset by one frame, target `B[t+15]`, stride 1 — adjacent samples are near-duplicates.

**The seven-fold split** [A]. `timeseries_train_test_split` uses `KFold(n_splits=7)` **without shuffling** and takes fold 4 as a contiguous test block. The preprint describes the same logic for hyper-parameter selection on Worm-1, explicitly to avoid iid sampling "due to high time correlations" [E]. Two caveats: `main.py` never calls the split — it trains on all windows and projects all windows, so the demo reports no held-out number; and seven contiguous folds of one 18-minute recording still share worm, day and session. Worm-held-out evaluation is strictly harder.

**Reported evaluation** [E]. The preprint reports dynamic prediction error `1 − MSE_model/MSE_io` against a copy-input autoregressor, behaviour decoding, and comparison with PCA, t-SNE, autoencoder, autoregressor-autoencoder and CEBRA-hybrid at 3 dimensions. Appendix A reports 94.3% decoding accuracy, with errors concentrated between forward and slowing. The code also contains `r2` and `hits_at_rank` helpers and a `markov_testing_model.h5` checkpoint, but no Markovianity test is defined in `functions.py`: "Markovian by construction" is an architectural constraint, not a test result.

**Limitations.** Deterministic latent dynamics — one path per initial condition, so branching in the embedding is geometry, not stochastic dynamics. Behaviour supervision shapes the geometry (10% of loss, shared encoder), so the latent is not neutral. The dimension is chosen for display. One dataset, no external validation, no perturbation.

---

## 4. NC-MCM: claims and code

**Framework.** Cognitive states are causally meaningful abstractions of neuronal states [E]. Behavioural causal consistency (BCC) requires a surjective map π such that any two neuronal states in the same pre-image induce the same `P(B|do(X=x))`; dynamic causal consistency (DCC) is the analogue for the next cognitive state. A triple of neuronal model, cognitive model and map satisfying both is an NC-MCM.

**Pipeline** (`nilab/src`) [A]:

1. **Behaviour probabilities.** `learn_mcm` fits `logreg` models; for eight behaviours `multiclasstrain` builds all 28 pairwise classifiers, `multiclasspredict` uses majority vote, and `bprob` stacks the 28 probabilities. The paper confirms 28 pairwise classifiers and notes that one-vs-all or intrinsic multiclass models would also be admissible [E].
2. **Cognitive states.** `kmeans_mcm` runs k-means in the 28-D probability space for k = 1…20 (20 values, matching the paper), 100 restarts per k, and for each k keeps the restart with the **largest** Markovianity p-value.
3. **Markovianity test** (`causal.jl`). Estimates `P(z_t|z_{t−1})` and `P(z_t|z_{t−1},z_{t−2})`, computes `T = Σ_states Var_{z_{t−2}} P(z_t|z_{t−1},z_{t−2})`, simulates 1 000 first-order chains from `P(z_t|z_{t−1})`, and returns an upper-tail p-value. Under a first-order chain the conditional variance across `z_{t−2}` is zero up to sampling noise, so large `T` is evidence against Markovianity. The published description matches [E].
4. **Reported outcome** [E]: at α = 0.05, cognitive trajectories with 3–19 states are not rejected, with the maximum average p-value at seven states. The behavioural chain itself is rejected (p = 0.015), which motivates the cognitive level.

**What is and is not licensed** [E]. With no interventional data, the model is built on `P(B|X)`, not `P(B|do(X))`: it is behaviourally *observationally* consistent, not behaviourally causally consistent, and the paper says so. DCC follows from Markovianity, so cognitive *dynamics* support causal-sounding statements while the cognitive→behaviour link does not. The perturbation vectors (mean neuronal state in `C1:revsus` minus `C3:revsus`, etc.) are presented as **empirically testable predictions** of external stimulation — exactly the epistemic level of a virtual perturbation. Assumptions: causal sufficiency/Markovianity of behaviourally relevant dynamics, faithfulness, time-homogeneity, stationarity, no instantaneous causation (they note this may fail at slow sampling), discrete states.

**Caveats visible in the code** [A]/[H]. (i) p-values are **maximised over 100 restarts per k**, making them extreme order statistics rather than valid p-values; a correct procedure would fix the clustering or nest selection inside the null. (ii) 5 worms × 20 values of k at α = 0.05 with no multiplicity correction. (iii) Accuracy printed by `learn_mcm` is in-sample, and the separate 10-fold `crossvalidate` defaults to `random_splits = true` on autocorrelated traces. (iv) Behaviour labels are deduced from AVA/SMD/RIB activity; excluding those eight neurons mitigates but does not eliminate circularity, since the labels were *defined* by a pattern other neurons may carry. (v) The "intrinsic randomness" inference assumes a latent effect would be detectable at the chosen abstraction and test power; a latent acting weakly or within a coarse state would be invisible. "Consistent with" is the correct reading, as the authors say.

---

## 5. BunDLe-Net vs. NC-MCM

Same group, same five worms, often mentioned together (NC-MCM ships BunDLe-Net embeddings as a comparison). They are not interchangeable.

| Axis | BunDLe-Net | NC-MCM |
|---|---|---|
| Target | Continuous 3-D embedding of a 15-frame window + latent transition map | Discrete partition of 28-D behaviour-probability space |
| Modelled | Neural dynamics *and* behaviour jointly | Behaviour probabilities only |
| Latent | Continuous, deterministic, autonomous `Y_{t+1} = Y_t + T_Y(Y_t)` | Discrete states with a transition matrix |
| Supervision | Behaviour label in the loss (10%) | Labels fit classifiers, then discarded |
| Output | Trajectory geometry (bundles, loops, apparent branch points) | Cognitive-state graph; perturbation vectors |
| Markov test | None in released code; constraint is architectural | Explicit surrogate-based test |
| Causal claim | None | DCC for cognitive dynamics; explicitly not BCC for behaviour |
| Failure mode | Trivial constant embedding; behaviour-driven geometry; deterministic branching | Clustering artefacts; optimised p-values; hidden latent confounders |

**Bottom line.** BunDLe-Net is a *representation learner*; NC-MCM is an *abstraction-and-test framework*. Neither estimates a stochastic generator, neither intervenes, and neither separates bifurcation from noise. NeuroBranch's contribution must live in that gap.

---

## 6. Biological dynamics

**Established** [E]. Kato et al. showed that much brain-wide activity is coordinated, that the global state traverses a cycle whose segments map onto motor commands, and that this assembles run-and-turn sequences. The no-stimulus subset has eight states — forward, slowing, dorsal turn, ventral turn, reversal-1, reversal-2, sustained reversal, ambiguous "no state" — whereas stimulated and AVA-HisCl datasets use four; "eight states" is a property of this subset, not the species [E/A].

**Artifact facts** from `WT_NoStim.mat` [A]: five worms; 109, 135, 131, 125, 129 traces; 3 021–3 311 frames; exactly 1 080 s each; fps 2.797–3.066; eight mutually exclusive state vectors summing to the frame count. Occupancy is heavily imbalanced — sustained reversal and slowing dominate; "no state" is 18–76 frames (0.6–2.5%). The repo audit adds that only 31–47 neurons per worm carry identities (23–38% of traces), with 192 distinct names across worms, and that behaviour labels derive from AVA/SMD/RIB activity.

Kaplan et al. extended this to a hierarchy: slow population dynamics toggle faster motor oscillations, nested again at shorter timescales [E]. Branching analysis should therefore expect decision points at more than one horizon. Atanas et al. provide the freely moving complement, with per-neuron encoding models and state-dependent encoding [E] — the natural external validation set, at the cost of harder alignment and partial identification.

**The behavioural null** [E]. Costa et al. show behaviour alone admits a high-fidelity Markov model over posture microstates, predictive across scales. A Markovian description of behaviour therefore does not require a neural latent, and a neural model that reproduces behaviour transitions has not thereby explained them. Any claim that neural branching *causes* behavioural branching must beat a behaviour-only Markov model, not just persistence. Brennan & Proekt come closest to the predictive goal: loop identity plus phase predicts future motor commands up to 30 s ahead across individuals [E] — a warning against over-parameterised latents and a benchmark on this exact dataset.

---

## 7. Latent and switching state-space models

Four modelling commitments, each fixing what "branching" can mean.

**Discrete switching with continuous dynamics.** rSLDS places a discrete mode with logistic transitions on per-mode linear dynamics [E] — the closest formal analogue of "one of several dynamical regimes", with an explicit transition matrix. Weakness: fixed mode count, linearity within modes; structurally the same limitation as NC-MCM's fixed k.

**Nonlinear latent state space.** Deep Markov models replace linear transitions with nonlinear Gaussian ones and amortised variational inference [E]. They model a latent confound explicitly — precisely what NC-MCM's Markovianity argument assumes away. A deep Markov fit that beats a Markovian latent on held-out activity is direct evidence against causal sufficiency at the observed resolution.

**Continuous time.** Neural ODEs define and integrate a vector field [E]; for 2.9 Hz imaging of sub-second dynamics this is conceptually right, but deterministic. Neural SDEs add diffusion and give the formal handle NeuroBranch needs: divergence surviving σ → 0 is deterministic, divergence vanishing with σ is noise [E].

**Single-trial autoencoders.** LFADS infers per-trial latents with a controller and Poisson emissions [E]; it is the strongest single-trial baseline in systems neuroscience but depends on trial alignment that continuous 18-minute recordings lack. Pseudo-trials (e.g. aligned to reversal onset) change the statistical question.

**Connectome-constrained models.** Lappalainen et al. fix the connectome and optimise only neuron/synapse parameters, predicting activity across 26 studies [E]. Morrison & Young do the analogue for the *C. elegans* premotor network: 15 core neurons, >100 drivers, cubic intrinsic dynamics, a ReLU synaptic term, and a diffusive gap-junction term `diag(W(u − uᵀ))`, constrained by imaging and voltage clamp [E/A]. Their model reproduces switching statistics, synchrony and higher-order statistics — the only body of work here that produces behaviour-like switching mechanistically, and therefore the natural place to look for a genuine bifurcation.

---

## 8. Perturbation

Leifer et al. established that single identified cells can be stimulated or inhibited in freely moving animals at ~30 µm and ~50 fps, with acute behavioural effects in motor, egg-laying and mechanosensory circuits [E] — the empirical standard for "intervention" in this system. The Kato release also contains AVA-HisCl recordings; the dataset name indicates AVA targeted with a histamine-gated chloride channel, but the silencing protocol was not read in full here [A].

Neither BunDLe-Net nor NC-MCM manipulates the system. NC-MCM's perturbation vectors are state differences, explicitly framed as predictions [E]. NeuroBranch should copy that framing: a virtual perturbation is a *model-dependent counterfactual*, valuable because it is falsifiable against a real perturbation, not because it substitutes for one. The gap is concrete — no public dataset combines cell-resolved whole-brain recording, tracked behaviour in freely moving animals, and targeted perturbation of identified neurons. Until one exists, causal claims rest on assumptions.

---

## 9. Causal inference and identifiability

`do(X=x)` replaces the structural equation for X and leaves the rest of the mechanism intact [E, Pearl 2009]. Observational `P(B|X)` equals `P(B|do(X))` only under conditions that must be argued: no unobserved confounding, correct conditioning on causal parents, positivity, and — for time series — no instantaneous causation and stable mechanisms [E, Peters et al. 2017; Runge et al. 2023].

Schölkopf et al. supply the key point for NeuroBranch: a low-dimensional representation is identifiable as a causal variable only with interventions, multiple environments, or specific inductive biases [E]. A 3-D BunDLe-Net embedding or a k-means partition is a *coordinate system*, not a causal variable. NC-MCM handles this by defining cognitive states through causal consistency rather than through the representation — a real advance, but one that inherits the assumption that the neuronal state space is causally sufficient for the behaviour.

Runge et al.'s checklist is directly usable [E]: state the assumed graph or sufficiency claim; test stationarity within recordings; test for instantaneous effects at the sampling resolution; treat unobserved confounders as the default hypothesis; validate on interventions or independent environments. The consequence is the claims ladder formalised in `RESEARCH_FOUNDATIONS.md`; most work here sits at levels 1–2, NC-MCM reaches roughly 2.5 for cognitive dynamics only, and nobody reaches level 4 for whole-brain *C. elegans*.

---

## 10. Metastability, branching, and the bifurcation confusion

**Three meanings, one word.**

1. **Critical branching** — each active unit triggers on average σ offspring; σ = 1 is critical and avalanche sizes become power laws [E, Beggs & Plenz 2003]. Subsampling drives estimated σ below 1 even in critical systems, and a corrected estimator exists [E, Wilting & Priesemann 2018].
2. **Trajectory branching** — a bundle of sampled trajectories splitting into distinguishable groups; the language BunDLe-Net uses for "a branching structure in the trajectories" and "probabilistic decisions… at specific bifurcation points" [E].
3. **Bifurcation** — a qualitative change in the phase portrait of a parameterised vector field as a parameter crosses a critical value (saddle-node, Hopf, pitchfork), detected by eigenvalue crossing or a change in the number/stability of invariant sets [E].

Conflating 2 with 3 is the most likely error in a project named "NeuroBranch".

**Why a branch point is not a bifurcation.** A bifurcation is a property of a *family* of systems indexed by a parameter, not of a single orbit. Along one orbit of a time-invariant autonomous system the future is determined by the state; trajectories cannot choose. Apparent splitting has other explanations, each testable: intrinsic noise (vanishes as σ → 0); an unobserved state (disappears when the latent is conditioned on or a richer history is used); deterministic expansion (Lyapunov spectrum, not the picture, is the evidence); parameter heterogeneity across worms or slow internal states; measurement and 3-D projection artefacts; or a genuine dynamic bifurcation, where a slow parameter crosses a critical value — the only case that earns the word, and it requires showing parameter dependence.

BunDLe-Net's text is careful enough to use "bifurcation points" descriptively; the risk is downstream reification. Morrison & Young's model is the right place to look for the real thing, because its parameters can be swept. A branch point should therefore be reported with its horizon, whether divergence survives noise reduction, whether it survives conditioning on richer history or a fitted latent, whether it recurs across worms, and whether a parameter sweep produces a qualitative change at the same location; otherwise the honest term is "decision point" (diagnostics in `RESEARCH_FOUNDATIONS.md` §5).

---

## 11. Evidence ledger

**Established [E].** A cyclical brain-wide state maps onto motor commands in immobilised *C. elegans*, with eight WT states (Kato 2015). Behavioural hierarchies are implemented by nested dynamics across three timescales (Kaplan 2020). Behaviour alone admits a high-fidelity Markov description (Costa 2024). Two macroscopic variables predict future motor commands up to 30 s ahead across individuals (Brennan & Proekt 2019). Connectome-constrained models reproduce premotor switching (Morrison & Young 2025) and predict fly visual activity across 26 studies (Lappalainen 2024). Targeted optogenetic perturbation is feasible in freely moving animals (Leifer 2011). Subsampling biases branching-ratio estimates, and a corrected estimator exists (Wilting & Priesemann 2018). A joint latent embedding with a residual transition map beats PCA/t-SNE/AE/CEBRA-style baselines on this dataset (Kumar 2023, preprint). Cognitive abstractions with 3–19 states are not rejected as Markovian, with a maximum average p-value at seven (Grosse-Wentrup 2024).

**Artifact-verified [A].** Kato file: 5 worms, 109/135/131/125/129 traces, 3 021–3 311 frames, 1 080 s, 2.797–3.066 fps, eight exclusive state vectors. Only 31–47 identified neurons per worm; 192 names across worms; labels derive from AVA/SMD/RIB (audit caveats). BunDLe-Net: MLP τ (50-30-25-10-d), residual T_Y, linear 8-way readout, γ = 0.9, paired 15-frame windows, contiguous `KFold(7)` with fold 4 as test, but `main.py` trains on all windows. NC-MCM: 28 pairwise logistic regressions, k-means 1–20 with 100 restarts, per-k maximum-p selection, 1 000 simulated chains. NeuroBranch: persistence, VAR(1), MLP, GRU, PCA-latent MLP and latent-kNN models; temporal and worm-holdout splits with a purge gap; R², bootstrap CIs, transition-distribution distance, rollout diagnostics. The `artifacts/smoke/*` runs are single-seed, few-epoch sanity checks (one reports negative one-step R² and behaviour accuracy far below the persistence baseline) and are **not** results.

**Hypotheses [H].** H1: much apparent branching in a 3-D embedding of Kato data will be explained by unobserved neurons and calcium smoothing rather than deterministic instability. H2: branch points will cluster at a few phase-space locations that recur across worms, more than a behaviour-only Markov model predicts. H3: a stochastic latent model will beat a deterministic latent of equal dimension on held-out one-step prediction, with the gain concentrated near branch points. H4: NC-MCM's Markovianity will weaken when clustering is fixed in advance rather than selected by maximal p-value. H5: a parameter sweep of the Morrison–Young model will show an eigenvalue crossing co-located with at least one recurrent branch point.

**Proposed contributions [P].** (1) A branching functional with a calibrated null (behaviour-Markov and persistence). (2) Noise and latent audits for every branch point. (3) A branch-versus-bifurcation diagnostic battery (Jacobian spectrum, parameter sweep, cross-worm recurrence). (4) Model-class-robust virtual perturbation across ≥3 model classes and multiple seeds, labelled as model-dependent. (5) An honest evaluation harness (blocked splits with purge, worm holdout, strong baselines, calibration, distributional metrics) — partly present already. (6) Reproduction of the two Vienna baselines as *reproduction*, with discrepancies documented.

---

## 12. Implications for NeuroBranch

1. **Do not call trajectory splits bifurcations** without a demonstrated parameter-dependent qualitative change; deterministic latents (BunDLe-Net's T_Y) cannot support branching claims at all.
2. **The strongest null is behavioural**, not persistence (Costa 2024), and **subsampling is not a detail** — ~1/3 of neurons observed means biased branching ratios.
3. **The Vienna methods are complements**: BunDLe-Net supplies a representation baseline, NC-MCM a discrete abstraction and Markov test. NeuroBranch's niche is the stochastic generator, the branch/bifurcation distinction, and perturbation-robustness analysis.
4. **Missing interventional ground truth should be stated, not hidden**; the package's `EPISTEMIC_WARNING` already encodes this.

---

## 13. Limitations of this review

- **No systematic search.** `web_search` was unavailable; sources came from a curated list resolved by DOI/arXiv/Europe PMC. Recent preprints and non-English work may be missing.
- **Abstract-level reading for most sources.** Full text was read only for BunDLe-Net and NC-MCM; for the rest, descriptions rely on abstracts, metadata and local code, and specific numbers are not restated.
- **No reproduction.** No model trained, no published metric recomputed; [A] statements are file inspections.
- **Kato full text not accessed**; the dataset readme, Europe PMC abstract and NC-MCM's description were used. Claims about the imaging protocol beyond "immobilised, 18 min, ~2.9 Hz, 8 states" are avoided.
- **Two preprints** (BunDLe-Net; the ML-method arXiv papers) should be re-checked for journal versions before submission.
- **URLs checked 2026-10-23.** All 16 DOIs were re-resolved through Crossref immediately before submission, with title, venue, year, volume and pages matched. The WormWideWeb dataset table (the source for Atanas 2023 recording counts) renders client-side, so row counts could not be confirmed from the page.
- **One expected item excluded.** No verifiable publication by D. Kovačević on *C. elegans* or causal neuroinformatics was found via Crossref author/affiliation queries; rather than cite something unverified, the collaboration is treated as programmatic context only.

---

## 14. Simulator-reset targeted verification (2026-10-31)

This is an addendum to the targeted review above, not a new systematic search.
Four identifiers that define the simulator's boundary were re-resolved through
Crossref immediately before the reset:

| Reference | Crossref result | Relevance to this phase |
|---|---|---|
| `10.1101/2023.08.08.551978` | Posted content, openRxiv, 2023-08-12; BunDLe-Net | Representation and transition baseline; no inverse decoder or intervention |
| `10.1371/journal.pcbi.1012674` | PLOS Computational Biology, 2024-12-16; a `new_version` update is recorded for 2025-01-09 | State abstraction and Markovianity test; not a stochastic generator |
| `10.1371/journal.pcbi.1013818` | PLOS Computational Biology, 2025-12-29; a `new_version` update is recorded for 2026-01-05 | Mechanistic premotor model; not fitted to the Kato five-worm subset |
| `10.1016/j.cell.2015.09.034` | Cell, 2015-10; Kato et al. | Primary observational dataset; no matched perturbation in `WT_NoStim` |

The Crossref records do not establish that any of these papers validates a
simulated named-neuron perturbation. BunDLe-Net remains a preprint in the
record returned here. The existence of a `new_version` relation is not a
retraction, correction, or expression of concern; it is a publisher update and
was not treated as a scientific change.

No claim in this repository depends on an unverified quotation attributed to
Denis Kovačević. The collaboration context remains programmatic.
