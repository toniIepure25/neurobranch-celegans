"""Build the compact supervisor-handoff evidence pack.

This script is intentionally read-only with respect to the frozen experiment
reports. It reads the final joint-dynamics tables and selected historical
summaries, then writes a compact synthesis under
``reports/simulator/supervisor_handoff``.

Usage:
    python scripts/build_supervisor_handoff.py
    python scripts/build_supervisor_handoff.py --manifest-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JOINT = ROOT / "reports" / "simulator" / "joint_dynamics"
OUT = ROOT / "reports" / "simulator" / "supervisor_handoff"
FIG = OUT / "figures"

WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION \u2014 NOT CAUSALLY VALIDATED."
STOP = "PROBABILISTIC CALIBRATION NOT ESTABLISHED"

COLORS = {
    "ink": "#17202a",
    "muted": "#5d6d7e",
    "grid": "#d5dbdb",
    "teal": "#117a65",
    "teal_light": "#a2d9ce",
    "amber": "#b9770e",
    "amber_light": "#f5cba7",
    "red": "#922b21",
    "red_light": "#f5b7b1",
    "gray": "#7f8c8d",
    "blue": "#1f618d",
    "paper": "#fbfcfc",
}

STATUS_COLORS = {
    "supported": COLORS["teal"],
    "exploratory": COLORS["amber"],
    "mixed": COLORS["amber"],
    "technical": COLORS["blue"],
    "negative": COLORS["red"],
    "withdrawn": COLORS["red"],
    "not_evaluable": COLORS["gray"],
}

MODEL_LABELS = {
    "J0_persistence": "J0 persistence",
    "J1_independent_diag": "J1 independent AR",
    "J2_population_diag": "J2 population diag",
    "J3_population_full": "J3 population full",
    "J4_independent_full": "J4 independent full",
    "J5_population_lead1_boot": "J5 population lead-1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _clean(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        return float(f"{float(value):.9g}")
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(JOINT / "score_summary.csv")
    paired = pd.read_csv(JOINT / "paired_scores.csv")
    secondary = pd.read_csv(JOINT / "secondary_paired.csv")
    calibration = pd.read_csv(JOINT / "calibration_h20.csv")
    return scores, paired, secondary, calibration


def phase_rows() -> list[dict[str, str]]:
    return [
        {
            "order": "1",
            "workstream": "Representation audit (BunDLe-Net / NC-MCM)",
            "question": "Is the published low-dimensional geometry a leak-free, generalizing representation?",
            "key_result": (
                "Distributed BunDLe-Net embeddings are transductive; kNN 0.949-0.984 on the "
                "artifact. A causal-filter reimplementation reached 0.407 and silhouette -0.064. "
                "NC-MCM Markov test ported within 0.023; training not reproduced."
            ),
            "status": "technical",
            "source": "docs/BUNDLENET_REPRODUCTION_AUDIT.md; docs/LITERATURE_REVIEW.md",
            "note": "Artifact audit, not a full BunDLe-Net reproduction.",
        },
        {
            "order": "2",
            "workstream": "Kato baseline dynamics",
            "question": "Do nonlinear or latent models beat persistence on held-out Kato sessions?",
            "key_result": (
                "Persistence is strong. The validation-selected mlp_residual had test free-running "
                "R2 -0.51; VAR was descriptively best at 0.53 but was not selected. No learned "
                "family beat persistence by median validation skill."
            ),
            "status": "negative",
            "source": "docs/SIMULATOR_RESULTS.md; docs/PHASE6_FINAL_SCIENTIFIC_AUDIT.md",
            "note": "Interpolation and free-running results differ; see methods reconciliation.",
        },
        {
            "order": "3",
            "workstream": "Randi intervention atlas",
            "question": "Can a response operator generalize to unseen stimulated targets?",
            "key_result": (
                "3,371 valid events, 110 animals, 259 targets. CRO pair Spearman 0.098 versus 0.054 "
                "for the baseline; +0.044 interval excludes zero but misses the +0.05 gate. "
                "Connectivity AUROC 0.509 versus 0.762 for a connectome ridge."
            ),
            "status": "negative",
            "source": "docs/PHASE6_SUPERVISOR_BRIEF.md; docs/INTERVENTION_DATASET_AUDIT.md",
            "note": "Positive direction, negative prespecified gate; no causal claim.",
        },
        {
            "order": "4",
            "workstream": "Behavioural branching and turn direction",
            "question": "Do trajectories branch more than null models, and does neural activity add turn-direction information?",
            "key_result": (
                "Branching shows an entropy deficit, not excess. Stochastic VAR TV 0.513 versus "
                "Markov 0.264 and marginal 0.245. Turn direction: population increment +0.012 "
                "[-0.017, +0.043]; SAAV -0.004 [-0.012, +0.003]."
            ),
            "status": "negative",
            "source": "docs/BRANCHING_STATISTICAL_AUDIT.md; docs/PHASE4_SUPERVISOR_BRIEF.md; docs/PHASE5_SUPERVISOR_BRIEF.md",
            "note": "Behaviour labels are decoded from a subset of the same neurons.",
        },
        {
            "order": "5",
            "workstream": "Free-running neural trajectory simulation",
            "question": "Can a fitted model roll forward from a measured state and reproduce trajectory statistics?",
            "key_result": (
                "G1 valid rollout and G4 perturbation interface pass. G2 free-running advantage, "
                "G3 branching and G5 specific perturbation fail. G6 biological validation is "
                "not evaluable."
            ),
            "status": "negative",
            "source": "docs/SIMULATOR_RESULTS.md; reports/simulator/gates.json",
            "note": "G1/G4 are software-contract gates, not scientific validity.",
        },
        {
            "order": "6",
            "workstream": "VAR and stochastic simulator",
            "question": "Does a stochastic VAR reproduce distributions better than deterministic or Markov baselines?",
            "key_result": (
                "No learned family had positive median validation skill. Stochastic VAR had TV 0.513 "
                "versus persistence 0.531, but Markov 0.264 and marginal 0.245 were better. "
                "Switching VAR over-produced entropy without improving TV."
            ),
            "status": "negative",
            "source": "docs/PHASE6_STOCHASTIC_BRANCHING.md; docs/PHASE6_FINAL_SCIENTIFIC_AUDIT.md",
            "note": "High entropy is not evidence of realistic branching.",
        },
        {
            "order": "7",
            "workstream": "External compatibility and Kato Stim pilot",
            "question": "Does the frozen pipeline replicate on an external constant-oxygen cohort?",
            "key_result": (
                "Kato Stim pilot: 0/7 sessions beat persistence at H20; median skill -0.602, mean "
                "-0.593, bootstrap 95% CI [-1.024, -0.241]. Frozen transfer was blocked by "
                "worm-specific trace panels; only per-animal refits were possible."
            ),
            "status": "exploratory",
            "source": "reports/external_validation/kato_stim_pilot_summary.json; docs/EXTERNAL_DATASET_COMPATIBILITY.md",
            "note": "Raw WT_Stim.mat is absent from the current checkout; independence undocumented.",
        },
        {
            "order": "8",
            "workstream": "Latent and compositional dynamics",
            "question": "Do context-conditioned latent LDS or compositional operators improve autonomous rollout?",
            "key_result": (
                "Latent LDS H20 mean skill -1.616; stable VAR -1.481; selected 2-operator "
                "compositional model -1.771. All five sessions negative at H20. Coefficients "
                "collapsed toward a narrow policy."
            ),
            "status": "negative",
            "source": "docs/SIMULATOR_CONTEXT_RESULTS.md",
            "note": "Estimator failure on this data, not evidence against biological composition.",
        },
        {
            "order": "9",
            "workstream": "Adaptive VAR and non-stationarity",
            "question": "Does a rolling or exponentially weighted population VAR improve prospective H20 forecasting?",
            "key_result": (
                "Rolling w500 reduces paired H20 MSE by 0.657 (33.9%; 95% CI [0.326, 0.986]) and "
                "improves 86.3% of landmarks; 4/5 sessions improve. Distributional fidelity fails, "
                "and the corrected calcium rollout beats persistence in only one session."
            ),
            "status": "mixed",
            "source": "docs/ADAPTIVE_DYNAMICS_RESULTS.md",
            "note": "Deterministic MSE gain; not a calibrated distribution.",
        },
        {
            "order": "10",
            "workstream": "Probabilistic adaptive calibration",
            "question": "Does the adaptive VAR support a calibrated stochastic simulator?",
            "key_result": (
                "No. Primary H20 energy 4.605 versus local mean 3.263 and independent AR 3.077; "
                "coverage 0.904/0.992/0.999; spread/error 14.85. The stochastic extension was "
                "grossly overdispersed."
            ),
            "status": "negative",
            "source": "docs/PROBADAPT_RESULTS.md",
            "note": "The deterministic mean gain did not survive the stochastic extension.",
        },
        {
            "order": "11",
            "workstream": "Corrected lead-1 innovations",
            "question": "Was the overdispersion caused by the innovation pool, and can it be corrected?",
            "key_result": (
                "Historical block model energy 4.609, 95% coverage 0.999, reproduced to 5.5e-9. "
                "Corrected lead-1 population model energy 2.700, full-neuron 95% coverage 0.947, "
                "but 50% coverage 0.588. Pool construction is the dominant cause."
            ),
            "status": "exploratory",
            "source": "docs/UNCAL_RESULTS.md; docs/JOINTDYN_REPRODUCTION.md",
            "note": "Near-unit-root operator also amplifies noise; correction is necessary, not sufficient.",
        },
        {
            "order": "12",
            "workstream": "Joint-dynamics calibration",
            "question": "Does the corrected population model add joint information beyond independent AR?",
            "key_result": (
                "Yes in the fixed 10-D PCA: H20 energy +0.446 [0.183, 0.660], trajectory +0.375 "
                "[0.218, 0.522] versus independent AR. J2 replicates (energy 2.700, trajectory "
                "2.058). Calibration and covariance gates fail."
            ),
            "status": "exploratory",
            "source": "reports/simulator/joint_dynamics/paired_scores.csv; docs/JOINTDYN_RESULTS.md",
            "note": "Test split reused; five sessions are not five animals; exploratory.",
        },
    ]


def claim_rows() -> list[dict[str, str]]:
    return [
        {
            "id": "C1",
            "claim": "Distributed BunDLe-Net embeddings are transductive, not a leak-free generalization result.",
            "status": "supported",
            "estimate": "kNN 0.949-0.984; silhouette 0.447-0.565",
            "scope": "artifact audit, five distributed embeddings",
            "source": "docs/BUNDLENET_REPRODUCTION_AUDIT.md",
            "caveat": "Does not invalidate the published method; invalidates this artifact as a baseline.",
        },
        {
            "id": "C2",
            "claim": "A causal-filter BunDLe-Lite reimplementation does not reproduce the distributed geometry.",
            "status": "exploratory",
            "estimate": "kNN 0.407; silhouette -0.064",
            "scope": "one session, reduced training",
            "source": "docs/BUNDLENET_REPRODUCTION_AUDIT.md",
            "caveat": "Architecture-compatible only; not a full reproduction.",
        },
        {
            "id": "C3",
            "claim": "The NC-MCM Markov test was ported, but its training was not reproduced.",
            "status": "supported",
            "estimate": "p-value agreement within 0.023; 1,000 surrogates",
            "scope": "test statistic only",
            "source": "docs/LITERATURE_REVIEW.md",
            "caveat": "Agreement of one statistic is not a full reproduction.",
        },
        {
            "id": "C4",
            "claim": "Persistence is a strong baseline for Kato population dynamics.",
            "status": "supported",
            "estimate": "selected MLP test R2 -0.510; VAR 0.533",
            "scope": "Kato NoStim, blocked test, free-running",
            "source": "docs/SIMULATOR_RESULTS.md",
            "caveat": "Metric/split dependent; later adaptive VAR can beat persistence on MSE.",
        },
        {
            "id": "C5",
            "claim": "Observed trajectories do not show excess branching over null models.",
            "status": "negative",
            "estimate": "TV 0.513 simulator vs 0.264 Markov vs 0.245 marginal",
            "scope": "Kato NoStim, 20-frame conditional outcomes",
            "source": "docs/PHASE6_STOCHASTIC_BRANCHING.md",
            "caveat": "Not evidence that no biological decision points exist; this analysis does not establish them.",
        },
        {
            "id": "C6",
            "claim": "Randi optogenetic responses have low single-animal and repeat reliability.",
            "status": "supported",
            "estimate": "single 0.162 [0.134, 0.191]; repeat 0.243 [0.206, 0.279]",
            "scope": "126-152 targets, 110 animals, descriptive",
            "source": "docs/CLAIMS.md; docs/PHASE10_RESULTS.md",
            "caveat": "Targets share animals; intervals are descriptive.",
        },
        {
            "id": "C7",
            "claim": "The held-out-target response operator misses its prespecified effect gate.",
            "status": "negative",
            "estimate": "+0.044 [+0.028, +0.059] vs +0.05 gate; calibration outside target",
            "scope": "Randi, unseen targets, five seeds",
            "source": "docs/PHASE6_SUPERVISOR_BRIEF.md",
            "caveat": "Positive direction; no causal connectivity (AUROC 0.509 vs 0.762).",
        },
        {
            "id": "C8",
            "claim": "The example named-neuron virtual perturbation is not specific.",
            "status": "withdrawn",
            "estimate": "RIVL +2 SD Delta P(TURN)=+0.333; 4/19 controls >=0.333; p~0.21",
            "scope": "Kato NoStim, model-state edit",
            "source": "docs/INTERVENTION_ASSUMPTIONS.md",
            "caveat": "Biological validation NOT_EVALUABLE; corrected controls only.",
        },
        {
            "id": "C9",
            "claim": "The free-running simulator has no demonstrated dynamical or perturbation-specific advantage.",
            "status": "negative",
            "estimate": "G2/G3/G5 fail; G1/G4 pass; G6 not evaluable",
            "scope": "Kato NoStim, frozen gates",
            "source": "docs/SIMULATOR_RESULTS.md",
            "caveat": "G1/G4 are software-contract gates.",
        },
        {
            "id": "C10",
            "claim": "The Kato Stim constant-oxygen external pilot is negative.",
            "status": "exploratory",
            "estimate": "0/7 positive; median skill -0.602; mean -0.593 [-1.024, -0.241]",
            "scope": "seven sessions, first 360 s, per-animal refit",
            "source": "reports/external_validation/kato_stim_pilot_summary.json",
            "caveat": "Independence undocumented; raw file absent; not an oxygen-response test.",
        },
        {
            "id": "C11",
            "claim": "Adaptive rolling VAR improves deterministic H20 MSE over frozen VAR.",
            "status": "supported",
            "estimate": "+0.657 [0.326, 0.986]; 33.9%; 86.3% landmarks",
            "scope": "adaptive repeated-session reanalysis",
            "source": "docs/ADAPTIVE_DYNAMICS_RESULTS.md",
            "caveat": "Not distributional; session 2 slightly worse; not animal-independent.",
        },
        {
            "id": "C12",
            "claim": "Historical stochastic overdispersion was caused mainly by pooling multi-step residuals.",
            "status": "supported",
            "estimate": "energy 4.609 -> 2.700; 95% coverage 0.999 -> 0.948",
            "scope": "Kato NoStim, 175 landmarks, H20",
            "source": "docs/UNCAL_RESULTS.md",
            "caveat": "Near-unit-root operator also amplifies noise.",
        },
        {
            "id": "C13",
            "claim": "The population conditional mean improves joint H20 prediction over independent AR.",
            "status": "exploratory",
            "estimate": "energy +0.446 [0.183, 0.660]; trajectory +0.375 [0.218, 0.522]",
            "scope": "fixed 10-D PCA, 175 landmarks, 5 sessions",
            "source": "reports/simulator/joint_dynamics/paired_scores.csv",
            "caveat": "Test reused; overlapping landmarks; sessions are not animals.",
        },
        {
            "id": "C14",
            "claim": "Full innovation covariance does not help and harms joint diagnostics.",
            "status": "negative",
            "estimate": "main effect -0.086 energy; J3 covariance error 0.762",
            "scope": "2x2 factorial, H20, 10-D PCA",
            "source": "reports/simulator/joint_dynamics/factorial_decomposition.csv",
            "caveat": "Sample-size and estimator specific; not a general claim about covariance.",
        },
        {
            "id": "C15",
            "claim": "Marginal per-neuron CRPS does not improve reliably.",
            "status": "negative",
            "estimate": "+0.0048 [-0.0083, 0.0157]",
            "scope": "full-neuron CRPS, H20, 175 landmarks",
            "source": "reports/simulator/joint_dynamics/secondary_paired.csv",
            "caveat": "Interval includes zero; not proof of equivalence.",
        },
        {
            "id": "C16",
            "claim": "The all-neuron energy advantage is unresolved.",
            "status": "exploratory",
            "estimate": "+0.203 [-0.010, 0.442]",
            "scope": "all-neuron energy, H20, 175 landmarks",
            "source": "reports/simulator/joint_dynamics/secondary_paired.csv",
            "caveat": "Do not generalize the 10-D PCA result to all neurons.",
        },
        {
            "id": "C17",
            "claim": "Probabilistic calibration is not established.",
            "status": "negative",
            "estimate": "50/80/95 coverage 0.588/0.849/0.947; covariance 0.604 > 0.50",
            "scope": "J5, full-neuron and 10-D PCA, H20",
            "source": "reports/simulator/joint_dynamics/gates.json",
            "caveat": "Student-t and parameter-uncertainty corrections are worse.",
        },
        {
            "id": "C18",
            "claim": "The 0.560 measurement ceiling is a bound.",
            "status": "withdrawn",
            "estimate": "R_obs 0.2606; R_22 0.3006; R_id 0.3501",
            "scope": "Phase 11/14/15 reliability estimands",
            "source": "docs/PHASE15_RESULTS.md",
            "caveat": "87 of 126 targets never reach 22 animals; not a calibrated interval or ranking.",
        },
    ]


def build_summary(
    scores: pd.DataFrame,
    paired: pd.DataFrame,
    secondary: pd.DataFrame,
    calibration: pd.DataFrame,
) -> dict:
    gates = json.loads((JOINT / "gates.json").read_text(encoding="utf-8"))
    sessions = pd.read_csv(JOINT / "session_paired_ci.csv")
    sessions = sessions[
        (sessions["model"] == "J5_population_lead1_boot")
        & (sessions["reference"] == "J1_independent_diag")
    ]
    block = pd.read_csv(JOINT / "bootstrap_block_sensitivity.csv")
    block = block[
        (block["model"] == "J5_population_lead1_boot")
        & (block["reference"] == "J0_persistence")
        & (block["metric"] == "energy_h20")
    ]

    score_rows = []
    for _, row in scores.iterrows():
        score_rows.append(
            {
                "model": row["model"],
                "label": MODEL_LABELS.get(row["model"], row["model"]),
                "n": int(row["n"]),
                "energy_h20": _clean(row["energy_h20"]),
                "trajectory_h20": _clean(row["trajectory_energy_h20"]),
                "crps_h20_full": _clean(row["crps_h20_full"]),
                "coverage_full_50": _clean(row["coverage_h20_full_50"]),
                "coverage_full_80": _clean(row["coverage_h20_full_80"]),
                "coverage_full_95": _clean(row["coverage_h20_full_95"]),
                "coverage_pca_50": _clean(row["coverage_50"]),
                "coverage_pca_80": _clean(row["coverage_80"]),
                "coverage_pca_95": _clean(row["coverage_95"]),
                "covariance_error": _clean(row["neural_cov_error"]),
                "autocorr_error": _clean(row["autocorr_error"]),
                "width_95": _clean(row["width_95"]),
            }
        )

    def paired_row(model: str, reference: str, metric: str) -> dict:
        match = paired[
            (paired["model"] == model)
            & (paired["reference"] == reference)
            & (paired["metric"] == metric)
        ].iloc[0]
        return {
            "model": model,
            "reference": reference,
            "metric": metric,
            "n_pairs": int(match["n_pairs"]),
            "improvement": _clean(match["mean_improvement"]),
            "ci_low": _clean(match["ci_low"]),
            "ci_high": _clean(match["ci_high"]),
            "fraction_better": _clean(match["fraction_better"]),
        }

    paired_rows = [
        paired_row("J5_population_lead1_boot", "J1_independent_diag", "energy_h20"),
        paired_row("J5_population_lead1_boot", "J1_independent_diag", "trajectory_energy_h20"),
        paired_row("J5_population_lead1_boot", "J0_persistence", "energy_h20"),
        paired_row("J5_population_lead1_boot", "J0_persistence", "trajectory_energy_h20"),
        paired_row("J2_population_diag", "J1_independent_diag", "energy_h20"),
        paired_row("J2_population_diag", "J1_independent_diag", "trajectory_energy_h20"),
        paired_row("J3_population_full", "J1_independent_diag", "energy_h20"),
        paired_row("J3_population_full", "J1_independent_diag", "trajectory_energy_h20"),
    ]

    secondary_rows = []
    for _, row in secondary.iterrows():
        secondary_rows.append(
            {
                "model": row["model"],
                "reference": row["reference"],
                "metric": row["metric"],
                "improvement": _clean(row["mean_improvement"]),
                "ci_low": _clean(row["ci_low"]),
                "ci_high": _clean(row["ci_high"]),
                "fraction_better": _clean(row["fraction_better"]),
            }
        )

    calibration_rows = []
    for _, row in calibration.iterrows():
        calibration_rows.append(
            {
                "model": row["model"],
                "label": MODEL_LABELS.get(row["model"], row["model"]),
                "full_50": _clean(row["coverage_full_50"]),
                "full_80": _clean(row["coverage_full_80"]),
                "full_95": _clean(row["coverage_full_95"]),
                "pca_50": _clean(row["coverage_50"]),
                "pca_80": _clean(row["coverage_80"]),
                "pca_95": _clean(row["coverage_95"]),
                "width_95": _clean(row["width_95"]),
                "pit_error": _clean(row["pit_calibration_error"]),
            }
        )

    return {
        "schema_version": 1,
        "title": (
            "NeuroBranch: Evaluating Generative Neural Population Dynamics and "
            "Model-Dependent Perturbations in C. elegans"
        ),
        "generated_for": "MSc discussion with Denis Kovacevic, University of Vienna",
        "evidence_tier": "exploratory repeated-session reanalysis",
        "status": {
            "probabilistic": STOP,
            "biological_validation": "NOT_EVALUABLE",
            "warning": WARNING,
            "history_policy": "additive only; no amend, rebase, force-push or reset",
        },
        "data": {
            "primary": "Kato et al. 2015 WT_NoStim",
            "sessions": 5,
            "frames": 15662,
            "predictors": "103 per session (one session 102)",
            "landmarks": 175,
            "landmark_stride": 10,
            "primary_horizon": 20,
            "representation": "10-D PCA fitted on training frames only; 54.7%-71.0% variance",
            "ensemble_members": 100,
            "animal_independence": "not documented; five repeated sessions are not five animals",
        },
        "central_claim": {
            "short": (
                "The corrected population conditional mean improves joint H20 prediction "
                "over independent AR in a fixed 10-D PCA."
            ),
            "long": (
                "In a fixed 10-D PCA fitted on training frames only, the corrected "
                "population-mean VAR provides evidence of improved joint neural-trajectory "
                "prediction over an independent-neuron AR baseline at H20. The gain is a "
                "conditional-mean gain; calibration and intervention validity remain unresolved."
            ),
            "status": "exploratory, internally supported, not confirmatory",
            "primary": paired_rows[0],
            "trajectory": paired_rows[1],
            "scores": {
                "J5_energy": 2.7001496961975793,
                "J1_energy": 3.1457128799856453,
                "J5_trajectory": 2.059962140668421,
                "J1_trajectory": 2.435121220691635,
                "J2_energy": 2.700283360297526,
                "J2_trajectory": 2.0576276640958344,
            },
            "why_not_confirmatory": [
                "The test split has been reused across phases.",
                "The 175 landmarks overlap and are not independent samples.",
                "The five sessions are repeated recordings, not documented independent animals.",
                "The bootstrap resamples landmarks, not animals.",
                "The model choice was informed by earlier test results.",
            ],
        },
        "scores": score_rows,
        "paired": paired_rows,
        "secondary": secondary_rows,
        "calibration": calibration_rows,
        "sessions": [
            {
                "session": int(row["worm_id"]),
                "improvement": _clean(row["mean_improvement"]),
                "ci_low": _clean(row["ci_low"]),
                "ci_high": _clean(row["ci_high"]),
            }
            for _, row in sessions.iterrows()
        ],
        "block_sensitivity": [
            {
                "block": int(row["block"]),
                "improvement": _clean(row["mean_improvement"]),
                "ci_low": _clean(row["ci_low"]),
                "ci_high": _clean(row["ci_high"]),
            }
            for _, row in block.iterrows()
        ],
        "gates": gates,
        "factorial": {
            "mean_effect_energy": 0.220864,
            "full_covariance_effect_energy": -0.086097,
            "interaction_energy": 0.449131,
            "mean_effect_trajectory": 0.229738,
            "full_covariance_effect_trajectory": -0.071183,
            "interaction_trajectory": 0.295512,
        },
        "reconciliations": [
            (
                "Earlier GENFAIL free-running failure and later adaptive MSE gains use different "
                "frozen models, splits and metrics; neither overturns the other."
            ),
            (
                "J5 and J2 are effectively identical; the mean advantage is replicated across "
                "block-bootstrap and parametric diagonal innovations."
            ),
            (
                "The 0.560 reliability value is withdrawn as a bound; Phase 15 reports three "
                "different estimands and no calibrated interval."
            ),
            (
                "Kato Stim was analysed historically but WT_Stim.mat is absent from this checkout; "
                "the pilot is exploratory and cannot be rerun without re-downloading."
            ),
        ],
        "limitations": [
            "No documented independent animals; five repeated sessions.",
            "Reused test split; exploratory inference.",
            "Overlapping landmarks; effective sample smaller than 175.",
            "One fixed 10-D evaluation representation.",
            "Post-hoc model choice after earlier test results.",
            "50% coverage fails; covariance fails.",
            "No biological intervention; virtual edits are model-state edits.",
            "Behaviour labels are decoded from the same neural population.",
            "External Kato Stim pilot: seven sessions, undocumented independence, raw file absent.",
        ],
        "warning": WARNING,
    }


def save_tables() -> None:
    pd.DataFrame(phase_rows()).to_csv(OUT / "phase_timeline.csv", index=False)
    pd.DataFrame(claim_rows()).to_csv(OUT / "claim_ledger.csv", index=False)


def wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width))


def fig_timeline() -> None:
    rows = phase_rows()
    fig, ax = plt.subplots(figsize=(13.2, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.axis("off")
    ax.set_facecolor(COLORS["paper"])
    fig.patch.set_facecolor(COLORS["paper"])
    ax.plot([0.265, 0.265], [0, len(rows) - 1], color=COLORS["grid"], lw=3, zorder=0)
    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        color = STATUS_COLORS.get(row["status"], COLORS["gray"])
        if i % 2 == 0:
            ax.add_patch(
                FancyBboxPatch(
                    (0.005, y - 0.43),
                    0.99,
                    0.86,
                    boxstyle="round,pad=0.006,rounding_size=0.01",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                    zorder=0,
                )
            )
        ax.scatter([0.265], [y], s=150, color=color, edgecolor="white", lw=2, zorder=3)
        ax.text(
            0.012,
            y + 0.16,
            f"{row['order']}. {row['workstream']}",
            fontsize=8.9,
            weight="bold",
            color=COLORS["ink"],
            va="top",
        )
        ax.text(
            0.012,
            y - 0.05,
            wrap(row["question"], 47),
            fontsize=7.0,
            color=COLORS["muted"],
            va="top",
            linespacing=1.2,
        )
        ax.text(
            0.285,
            y + 0.16,
            wrap(row["key_result"], 86),
            fontsize=7.0,
            color=COLORS["ink"],
            va="top",
            linespacing=1.25,
        )
        label = row["status"].replace("_", " ").upper()
        ax.text(
            0.985,
            y + 0.30,
            label,
            fontsize=6.4,
            color="white",
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.28", fc=color, ec="none"),
        )
    ax.text(
        0.012,
        len(rows) - 0.18,
        "CHRONOLOGICAL EVIDENCE LADDER",
        fontsize=12,
        weight="bold",
        color=COLORS["ink"],
        va="top",
    )
    ax.text(
        0.285,
        len(rows) - 0.18,
        "What was tested, what survived, and what did not",
        fontsize=9,
        color=COLORS["muted"],
        va="top",
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "fig01_evidence_timeline.png", dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_central(scores: pd.DataFrame, paired: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 5.7), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )
    fig.patch.set_facecolor("white")
    palette = {
        "J0_persistence": COLORS["gray"],
        "J1_independent_diag": COLORS["amber"],
        "J2_population_diag": COLORS["teal"],
        "J3_population_full": COLORS["red"],
        "J4_independent_full": COLORS["blue"],
        "J5_population_lead1_boot": COLORS["teal"],
    }
    for _, row in scores.iterrows():
        model = row["model"]
        ax1.scatter(
            row["energy_h20"],
            row["trajectory_energy_h20"],
            s=110 if model in {"J5_population_lead1_boot", "J2_population_diag"} else 70,
            color=palette.get(model, COLORS["gray"]),
            edgecolor="white",
            lw=1.5,
            zorder=3,
        )
        dx, dy = 0.035, 0.018
        if model == "J2_population_diag":
            dy = 0.045
        if model == "J5_population_lead1_boot":
            dy = -0.055
        if model == "J0_persistence":
            dx, dy = 0.03, -0.04
        ax1.annotate(
            MODEL_LABELS.get(model, model),
            (row["energy_h20"], row["trajectory_energy_h20"]),
            xytext=(row["energy_h20"] + dx, row["trajectory_energy_h20"] + dy),
            fontsize=7.3,
            color=COLORS["ink"],
        )
    ax1.set_xlabel("H20 state energy (lower is better)", fontsize=9)
    ax1.set_ylabel("H20 trajectory energy (lower is better)", fontsize=9)
    ax1.set_title("Six models, same fixed 10-D PCA", fontsize=10.5, weight="bold", loc="left")
    ax1.grid(alpha=0.25, lw=0.7)
    ax1.set_axisbelow(True)
    ax1.text(
        0.02,
        0.97,
        "J2 and J5 overlap: mean advantage is not one sampler",
        transform=ax1.transAxes,
        fontsize=7.2,
        color=COLORS["muted"],
        va="top",
    )

    rows = [
        ("J5 vs J1  ·  energy", paired, "J5_population_lead1_boot", "J1_independent_diag", "energy_h20"),
        ("J5 vs J1  ·  trajectory", paired, "J5_population_lead1_boot", "J1_independent_diag", "trajectory_energy_h20"),
        ("J5 vs J0  ·  energy", paired, "J5_population_lead1_boot", "J0_persistence", "energy_h20"),
        ("J5 vs J0  ·  trajectory", paired, "J5_population_lead1_boot", "J0_persistence", "trajectory_energy_h20"),
        ("J2 vs J1  ·  energy", paired, "J2_population_diag", "J1_independent_diag", "energy_h20"),
        ("J2 vs J1  ·  trajectory", paired, "J2_population_diag", "J1_independent_diag", "trajectory_energy_h20"),
        ("J3 vs J1  ·  energy", paired, "J3_population_full", "J1_independent_diag", "energy_h20"),
        ("J3 vs J1  ·  trajectory", paired, "J3_population_full", "J1_independent_diag", "trajectory_energy_h20"),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (_label, frame, model, reference, metric) in zip(ys, rows, strict=True):
        row = frame[
            (frame["model"] == model)
            & (frame["reference"] == reference)
            & (frame["metric"] == metric)
        ].iloc[0]
        color = COLORS["teal"] if row["ci_low"] > 0 else COLORS["gray"]
        ax2.errorbar(
            row["mean_improvement"],
            y,
            xerr=[
                [row["mean_improvement"] - row["ci_low"]],
                [row["ci_high"] - row["mean_improvement"]],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=3,
            ms=6,
        )
        ax2.text(
            row["ci_high"] + 0.03,
            y,
            f"{row['mean_improvement']:+.3f}  [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]",
            fontsize=7.2,
            va="center",
            color=COLORS["ink"],
        )
    ax2.axvline(0, color=COLORS["red"], lw=1.2, ls="--")
    ax2.set_yticks(ys)
    ax2.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax2.set_xlim(-0.30, 1.18)
    ax2.set_xlabel("Improvement (reference minus model; positive favours first model)", fontsize=9)
    ax2.set_title("Paired H20 contrasts with 95% block-bootstrap intervals", fontsize=10.5, weight="bold", loc="left")
    ax2.grid(axis="x", alpha=0.25, lw=0.7)
    ax2.set_axisbelow(True)
    ax2.text(
        0.98,
        0.03,
        "Teal: interval excludes zero.  Gray: includes zero.",
        transform=ax2.transAxes,
        fontsize=7.0,
        color=COLORS["muted"],
        ha="right",
    )
    fig.tight_layout(pad=1.2)
    fig.savefig(FIG / "fig02_central_result.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_calibration(calibration: pd.DataFrame) -> None:
    models = [
        "J0_persistence",
        "J1_independent_diag",
        "J2_population_diag",
        "J3_population_full",
        "J5_population_lead1_boot",
    ]
    palette = {
        "J0_persistence": COLORS["gray"],
        "J1_independent_diag": COLORS["amber"],
        "J2_population_diag": COLORS["teal_light"],
        "J3_population_full": COLORS["red"],
        "J5_population_lead1_boot": COLORS["teal"],
    }
    nominal = np.array([0.50, 0.80, 0.95])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, prefix, title in zip(
        axes,
        ["coverage_full", "coverage"],
        ["Full-neuron coverage", "10-D PCA coverage"],
        strict=True,
    ):
        ax.plot([0.45, 1.0], [0.45, 1.0], ls="--", color=COLORS["ink"], lw=1.0, label="nominal")
        for model in models:
            row = calibration[calibration["model"] == model].iloc[0]
            vals = [row[f"{prefix}_50"], row[f"{prefix}_80"], row[f"{prefix}_95"]]
            ax.plot(
                nominal,
                vals,
                marker="o",
                ms=5,
                lw=2 if model == "J5_population_lead1_boot" else 1.3,
                color=palette[model],
                label=MODEL_LABELS[model],
                alpha=0.95,
            )
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.02)
        ax.set_xticks(nominal)
        ax.set_xticklabels(["50%", "80%", "95%"])
        ax.set_xlabel("Nominal coverage", fontsize=9)
        ax.set_title(title, fontsize=10.2, weight="bold", loc="left")
        ax.grid(alpha=0.25, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Empirical coverage", fontsize=9)
    axes[0].annotate(
        "J5 50%: 0.588",
        xy=(0.50, 0.588),
        xytext=(0.53, 0.70),
        fontsize=7.3,
        color=COLORS["teal"],
        arrowprops=dict(arrowstyle="->", color=COLORS["teal"], lw=1),
    )
    axes[1].annotate(
        "J5 50%: 0.546",
        xy=(0.50, 0.546),
        xytext=(0.54, 0.66),
        fontsize=7.3,
        color=COLORS["teal"],
        arrowprops=dict(arrowstyle="->", color=COLORS["teal"], lw=1),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index("nominal")] + [labels.index(MODEL_LABELS[m]) for m in models]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        ncol=6,
        fontsize=7.2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle(
        "Calibration fails at 50% and worsens with full covariance",
        x=0.01,
        ha="left",
        fontsize=11.5,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(FIG / "fig03_calibration.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_example() -> None:
    observed = pd.read_csv(JOINT / "example_observed.csv")
    bands = pd.read_csv(JOINT / "example_bands.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    fig.patch.set_facecolor("white")
    ax.plot(
        observed["step"],
        observed["observed_pc1"],
        color=COLORS["ink"],
        lw=2.4,
        marker="o",
        ms=3.2,
        label="observed continuation",
    )
    for model, color, lw, label in [
        ("J5_population_lead1_boot", COLORS["teal"], 2.0, "J5 population mean"),
        ("J1_independent_diag", COLORS["amber"], 1.6, "J1 independent AR"),
        ("J0_persistence", COLORS["gray"], 1.2, "J0 persistence"),
    ]:
        frame = bands[bands["model"] == model]
        ax.plot(frame["step"], frame["q50"], color=color, lw=lw, label=label)
        if model == "J5_population_lead1_boot":
            ax.fill_between(
                frame["step"],
                frame["q05"],
                frame["q95"],
                color=color,
                alpha=0.18,
                label="J5 90% ensemble band",
            )
    ax.set_xlabel("H20 step after the last observed frame", fontsize=9)
    ax.set_ylabel("PC1 of the fixed training PCA", fontsize=9)
    ax.set_title(
        "One prespecified example: observed continuation vs model forecast",
        fontsize=10.5,
        weight="bold",
        loc="left",
    )
    ax.grid(alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.2, frameon=False, ncol=2, loc="best")
    ax.text(
        0.01,
        0.02,
        "Session 0, landmark 2690; first prespecified test landmark, not selected by score.",
        transform=ax.transAxes,
        fontsize=6.8,
        color=COLORS["muted"],
    )
    fig.tight_layout(pad=1.0)
    fig.savefig(FIG / "fig04_example_trajectory.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_proposals() -> None:
    packages = [
        ("Data audit / adapter", [30, 30]),
        ("Frozen evaluation", [50, 45]),
        ("Uncertainty / controls", [35, 45]),
        ("Calibration / design", [25, 35]),
        ("Reproduction package", [20, 15]),
        ("Writing / supervision", [8, 4]),
    ]
    colors = ["#1f618d", "#117a65", "#b9770e", "#922b21", "#7f8c8d", "#5d6d7e"]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    fig.patch.set_facecolor("white")
    y = [1, 0]
    left = [0, 0]
    for (name, values), color in zip(packages, colors, strict=True):
        ax.barh(y, values, left=left, color=color, edgecolor="white", height=0.45, label=name)
        for yi, li, vi in zip(y, left, values, strict=True):
            if vi >= 15:
                ax.text(li + vi / 2, yi, str(vi), ha="center", va="center", color="white", fontsize=7.5)
        left = [left[0] + values[0], left[1] + values[1]]
    ax.set_yticks(y)
    ax.set_yticklabels(["A  Replicate population mean\n168 h", "B  Intervention-aware design\n174 h"], fontsize=8.5)
    ax.set_xlabel("Work-package hours (6 ECTS budget)", fontsize=9)
    ax.set_xlim(0, 190)
    ax.set_title("Two bounded 6-ECTS scopes, no architecture search", fontsize=10.5, weight="bold", loc="left")
    ax.grid(axis="x", alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout(pad=1.0)
    fig.savefig(FIG / "fig05_proposed_scope.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def build_provenance() -> dict:
    sources = [
        "docs/DATA_PROVENANCE.md",
        "docs/SIMULATOR_RESULTS.md",
        "docs/ADAPTIVE_DYNAMICS_RESULTS.md",
        "docs/GENFAIL_RESULTS.md",
        "docs/PHASE15_RESULTS.md",
        "docs/BUNDLENET_REPRODUCTION_AUDIT.md",
        "reports/external_validation/kato_stim_pilot_summary.json",
        "reports/phase15/summary.json",
        "reports/simulator/joint_dynamics/score_summary.csv",
        "reports/simulator/joint_dynamics/paired_scores.csv",
        "reports/simulator/joint_dynamics/gates.json",
    ]
    hashes = {p: sha256(ROOT / p) for p in sources if (ROOT / p).exists()}
    return {
        "schema_version": 1,
        "checked_on": "2026-11-02",
        "source_hashes": hashes,
        "discrepancies": [
            {
                "id": "kato_stim_availability",
                "finding": (
                    "WT_Stim.mat is not present in the checkout, but a completed seven-session "
                    "external pilot is documented in reports/external_validation."
                ),
                "resolution": (
                    "Report the pilot as exploratory and historical; it cannot be rerun without "
                    "re-downloading the raw file. Do not claim cross-stimulation was never analysed."
                ),
                "severity": "important",
            },
            {
                "id": "data_provenance_stim_sentence",
                "finding": (
                    "docs/DATA_PROVENANCE.md says no WT_Stim.mat was downloaded or analysed; later "
                    "external-validation work supersedes that statement for its own scope."
                ),
                "resolution": (
                    "Scope the DATA_PROVENANCE sentence to the original inventory. The handoff "
                    "records the later evidence rather than deleting it."
                ),
                "severity": "important",
            },
            {
                "id": "five_sessions_animals",
                "finding": (
                    "Kato NoStim has five recording sessions; the OSF metadata does not document "
                    "five independent animals."
                ),
                "resolution": "Use 'sessions' everywhere; no animal-independent replication claim.",
                "severity": "critical",
            },
            {
                "id": "genfail_vs_adaptive",
                "finding": (
                    "GENFAIL reports no positive mean full-neuron MSE skill; adaptive dynamics "
                    "reports a rolling VAR gain over frozen VAR and persistence."
                ),
                "resolution": (
                    "Different frozen/adaptive models, splits and metrics. The joint result is a "
                    "third estimand: 10-D PCA energy/trajectory, 175 landmarks. Neither earlier "
                    "result is overwritten."
                ),
                "severity": "important",
            },
            {
                "id": "j2_j5_equivalence",
                "finding": (
                    "J2 and J5 have nearly identical scores; the headline uses J5 but the mean "
                    "advantage is not unique to its innovation sampler."
                ),
                "resolution": "Report J2 as an internal replication and avoid 'J5 is uniquely best'.",
                "severity": "important",
            },
            {
                "id": "innovation_acf_summary",
                "finding": (
                    "The protocol gate uses median absolute ACF5 0.00886; the signed median is "
                    "0.00783. Both are below threshold."
                ),
                "resolution": "State the summary convention; the decision is unchanged.",
                "severity": "minor",
            },
            {
                "id": "test_reuse",
                "finding": (
                    "The Kato NoStim test block was inspected in earlier phases and reused in the "
                    "joint-dynamics experiment."
                ),
                "resolution": "All test-split claims are exploratory; no confirmatory significance.",
                "severity": "critical",
            },
            {
                "id": "phase15_ceiling",
                "finding": (
                    "Phase 11's 0.560 ceiling is not a bound; Phase 15 shows R(22) is extrapolated "
                    "for 87 of 126 targets."
                ),
                "resolution": "Use the three Phase 15 estimands and no target ranking.",
                "severity": "critical",
            },
        ],
        "unresolved": [
            "No documented animal IDs for the Kato NoStim sessions.",
            "No matched neuronal intervention dataset for the fitted model.",
            "No current-checkout raw WT_Stim file for re-running the external pilot.",
            "No independent cohort for the joint-dynamics representation.",
        ],
    }


def build_manifest() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        if path.suffix.lower() == ".npz":
            continue
        files[path.relative_to(OUT).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    documents = {}
    doc_names = [
        "HANDOFF_EXECUTIVE_SUMMARY.md",
        "HANDOFF_RESEARCH_MEMO.md",
        "HANDOFF_PRESENTATION.md",
        "HANDOFF_SCIENTIFIC_CLAIMS.md",
        "HANDOFF_METHODS_AND_LIMITATIONS.md",
        "HANDOFF_RESEARCH_PROPOSALS.md",
        "HANDOFF_QUESTIONS_FOR_DENIS.md",
        "HANDOFF_DEMO_GUIDE.md",
        "HANDOFF_REPRODUCIBILITY.md",
    ]
    for name in doc_names:
        path = ROOT / "docs" / name
        if path.exists():
            documents[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    sources = {}
    for path in [
        JOINT / "score_summary.csv",
        JOINT / "paired_scores.csv",
        JOINT / "secondary_paired.csv",
        JOINT / "calibration_h20.csv",
        JOINT / "gates.json",
        JOINT / "example_observed.csv",
        JOINT / "example_bands.csv",
    ]:
        if path.exists():
            sources[rel(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema_version": 1,
        "root": rel(OUT),
        "hash_algorithm": "sha256",
        "files": files,
        "documents": documents,
        "referenced_sources": sources,
        "excluded": [
            "raw data and checkpoints",
            "trajectory_examples.npz (optional 8.2 MB visualization subset)",
            "full per-landmark score tables",
            "manifest.json itself",
        ],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        manifest = build_manifest()
        print(f"wrote manifest with {len(manifest['files'])} files")
        return

    scores, paired, secondary, calibration = load_tables()
    summary = build_summary(scores, paired, secondary, calibration)
    save_tables()
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    provenance = build_provenance()
    (OUT / "provenance_audit.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig_timeline()
    fig_central(scores, paired)
    fig_calibration(calibration)
    fig_example()
    fig_proposals()
    build_manifest()
    print(f"wrote handoff pack to {rel(OUT)}")


if __name__ == "__main__":
    main()
