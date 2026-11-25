# Information and leakage contract

| Field | Learner during fitting | Predictor during scoring | Evaluation harness |
|---|---|---|---|
| Passive positions | Both positions in full conditions, current position only in compressed conditions | Same rule | Both positions and latent checkpoint |
| Sensor S | One pre-probe observation in sensor conditions. Masked in base conditions. | Same rule | Available for checking the controlled distinction |
| Context label and velocity | Neither is supplied as a latent column | Neither is supplied as a latent column | Available for exact truth and separate mechanism averages |
| Fixed action | Known as protocol semantics for the sole target | Same semantics without a varying action feature | Applies the specified probe |
| Intervention outcome Y | Exactly one realized target label per training episode | No outcome | Uses exact categorical response laws for scoring; no score Y is generated |
| Metadata | No episode ID, row index, namespace, seed, path, device or model label | No metadata | Stores reproducibility plan and post-score aggregate support summaries |

In the sensor conditions the measured S conveys exactly the information in the latent context by experimental design. The input projection exposes only the observation column. The base conditions mask it during fitting and scoring. All learners receive the same training episodes and labels; the declared compression transformation and sensor availability are the only assigned input differences. No extra passive-only trajectories, external datasets, calibration probes or test labels are used. This is supervised intervention-response prediction.

Generate context and initial conditions independently for every episode. Both mechanisms share one generator and data format, with random interleaving in draw order and identical trajectory lengths, missingness, preprocessing and fixed action. No mechanism-specific files, separate input RNGs or stratified row ordering are used. A table is keyed only by permitted feature tuples. Evaluation truth and mechanism labels remain in evaluator-only arrays. Tests perturb hidden fields without changing features or fits, change S to verify masking and availability, and check that score episodes carry no Y.

Use exact enumeration for all positive passive start states under the support diameter, even for the preserved 1e−8 fixture. Under sampling, report observed and missed positive states for each training prefix and evaluation block, together with omitted mass and the conservative 2δ upper bound on mass affected in a two-draw TV diagnostic. Mark when this bound exceeds 0.05. It is an omission flag, not an error correction or an exclusion criterion. The MSc support is exactly twelve states of mass 1/12. The calibrated rare fixture remains separate. Multi-step historical acceptance restricts support at the initial checkpoint and retains legal later states.
