"""Build the machine-readable master-review ledgers.

The numeric inputs are frozen project artifacts plus the independent checks in
``reports/master_review``.  This script does not refit a model and does not touch
the raw archive.  It exists so the claims, corrections, capacity audit and
quality gate can be regenerated and hash-stamped.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "master_review"
COMMON = ROOT / "reports" / "simulator" / "kato_common_state"
PER_LANDMARK = COMMON / "model_per_landmark.csv"
FACTOR_VARIANCE = COMMON / "factor_variance.csv"
STATS = REPORTS / "statistical_audit.json"
IDENTITY = REPORTS / "dataset_identity_audit.json"
BOUNDS = REPORTS / "frame_landmark_bounds.json"


def _write_json(name: str, payload: object) -> Path:
    path = REPORTS / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effect_sizes() -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(PER_LANDMARK)
    energy = frame.pivot_table(
        index=["session", "landmark"], columns="model", values="pca10_energy_h20"
    )
    mse = frame.pivot_table(
        index=["session", "landmark"], columns="model", values="pca10_mse_mean_h20"
    )
    diff = energy["B5_common_k10"] - energy["B3_population"]
    mse_diff = mse["B5_common_k10"] - mse["B3_population"]
    quartile = pd.qcut(energy["B3_population"], 4, labels=False)
    coverage = (
        frame[frame["model"].isin(["B3_population", "B5_common_k10"])]
        .groupby("model")[["pca10_coverage_50", "pca10_coverage_80", "pca10_coverage_95"]]
        .mean()
        .to_dict(orient="index")
    )
    payload = {
        "n_pairs": int(len(diff)),
        "energy": {
            "mean": float(diff.mean()),
            "sd": float(diff.std(ddof=1)),
            "median": float(diff.median()),
            "iqr": [float(diff.quantile(0.25)), float(diff.quantile(0.75))],
            "standardized_mean_dz_descriptive": float(diff.mean() / diff.std(ddof=1)),
            "fraction_positive": float((diff > 0).mean()),
        },
        "mse": {
            "mean": float(mse_diff.mean()),
            "sd": float(mse_diff.std(ddof=1)),
            "median": float(mse_diff.median()),
            "standardized_mean_dz_descriptive": float(
                mse_diff.mean() / mse_diff.std(ddof=1)
            ),
            "fraction_positive": float((mse_diff > 0).mean()),
        },
        "energy_by_b3_energy_quartile": [
            {
                "quartile": int(q),
                "n": int((quartile == q).sum()),
                "mean_advantage": float(diff[quartile == q].mean()),
            }
            for q in range(4)
        ],
        "corr_advantage_with_b3_energy": float(
            np.corrcoef(diff.to_numpy(), energy["B3_population"].to_numpy())[0, 1]
        ),
        "coverage_at_nominal_95": {
            model: {
                "50": float(values["pca10_coverage_50"]),
                "80": float(values["pca10_coverage_80"]),
                "95": float(values["pca10_coverage_95"]),
            }
            for model, values in coverage.items()
        },
        "note": (
            "The energy effect is small (dz 0.27); the mean-MSE effect is negligible "
            "(dz 0.04). B5's 95% coverage is 0.725 versus B3's 0.938, so the energy "
            "contrast is partly a spread/calibration contrast."
        ),
    }
    return diff, payload


def _capacity(identity: dict[str, object], diff: pd.DataFrame, effect: dict[str, object]) -> dict[str, object]:
    factors = pd.read_csv(FACTOR_VARIANCE)
    k10 = factors[factors["k"] == 10].set_index("session")[
        "explained_variance_fraction"
    ]
    rows = []
    for rec in identity["recordings"]:
        n = int(rec["predictor_dimension_after_label_exclusion"])
        session = int(rec["session_index"])
        b3 = n * (n + 1)
        b5 = 12 * n + 110  # 12 per neuron (own lag + 10 loadings + intercept) + 10x11 factor VAR
        rows.append(
            {
                "session": session,
                "native_predictors": n,
                "b3_parameters": b3,
                "b5_k10_parameters": b5,
                "b3_over_b5_ratio": b3 / b5,
                "b5_k10_training_variance_fraction": float(k10.loc[session]),
                "b5_k10_residual_variance_fraction": float(1.0 - k10.loc[session]),
            }
        )
    ratios = [row["b3_over_b5_ratio"] for row in rows]
    payload = {
        "definition": (
            "B3: n(n+1) coefficients including unpenalized intercept. B5-10: 12n "
            "(own lag, ten loadings, intercept) + 10x11 factor-VAR coefficients."
        ),
        "per_session": rows,
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "b5_training_variance_fraction_min": float(min(r["b5_k10_training_variance_fraction"] for r in rows)),
        "b5_training_variance_fraction_max": float(max(r["b5_k10_training_variance_fraction"] for r in rows)),
        "b5_residual_variance_fraction_max": float(max(r["b5_k10_residual_variance_fraction"] for r in rows)),
        "b3_vs_b4m_energy_gain": 0.539517388834083,
        "common_factor_increment": 0.27507939616572663,
        "residual_increment": 0.2644379926683563,
        "ordered_shares": {
            "baseline_B4m": {"common": 0.5098619652652594, "residual": 0.49013803473474066},
            "baseline_B1m": {"common": 0.235, "residual": 0.765},
            "baseline_B0": {"common": 0.724, "residual": 0.276},
        },
        "conclusion": (
            "B3 has 7.96-9.95x more parameters than B5-10 and arbitrary residual "
            "coordinates that B5 cannot represent. The ordered 51/49 split is "
            "arithmetic, not a structural or biological partition."
        ),
    }
    return payload


def _claims(effect: dict[str, object]) -> dict[str, object]:
    source = "reports/master_review"
    claims = [
        {
            "id": "D01",
            "claim": "The analysed file is Kato WT_NoStim with SHA256 matching the OSF object.",
            "status": "LIMITED",
            "claim_type": "archive fact",
            "evidence": [
                f"{source}/dataset_identity_audit.json",
                "https://api.osf.io/v2/nodes/2395t/files/osfstorage/",
            ],
            "why": "Direct hash and byte-size verification; tier vocabulary has no VERIFIED label, so LIMITED means bounded to the archive, not uncertain.",
        },
        {
            "id": "D02",
            "claim": "Five recordings contain 15,662 frames and 592 behaviour-label-excluded predictors in total.",
            "status": "LIMITED",
            "claim_type": "descriptive",
            "evidence": [f"{source}/dataset_identity_audit.json"],
            "why": "Direct HDF5 count; predictor count excludes eight label-defining neuron names.",
        },
        {
            "id": "D03",
            "claim": "The five entries are five distinct animals, not five repeated sessions of one preparation.",
            "status": "LIMITED",
            "claim_type": "identity inference",
            "evidence": [
                f"{source}/dataset_identity_audit.json",
                "https://doi.org/10.1016/j.cell.2015.09.034",
                "https://doi.org/10.1371/journal.pcbi.1012674",
            ],
            "why": "Publication and secondary descriptions report five worms; OSF has no animal key. One acquisition per animal is inferred, not keyed. They are one laboratory cohort, not independent replication.",
        },
        {
            "id": "D04",
            "claim": "The eight behaviour labels are independent external measurements.",
            "status": "INVALIDATED",
            "claim_type": "circularity",
            "evidence": ["docs/DATA_PROVENANCE.md", "data/raw/kato2015/readme_Kato2015.txt"],
            "why": "Labels derive from command/interneuron calcium and behaviour; they are not independent of the recorded brain.",
        },
        {
            "id": "D05",
            "claim": "There are 35 H100 test landmarks in every recording.",
            "status": "INVALIDATED",
            "claim_type": "documentation",
            "evidence": [f"{source}/frame_landmark_bounds.json", "docs/ADAPTIVE_DYNAMICS_MODELS.md"],
            "why": "Actual counts are 35, 35, 34, 38, 33 (175 total).",
        },
        {
            "id": "D06",
            "claim": "All recordings share the test boundary 2689:3137.",
            "status": "INVALIDATED",
            "claim_type": "documentation",
            "evidence": [f"{source}/frame_landmark_bounds.json", "docs/HANDOFF_METHODS_AND_LIMITATIONS.md"],
            "why": "2689:3137 is session 0 only; the code uses per-recording boundaries.",
        },
        {
            "id": "D07",
            "claim": "Every reported forecast target lies inside its recording and inside the test mask, with no padding or interpolation.",
            "status": "SUPPORTED_EXPLORATORY",
            "claim_type": "technical audit",
            "evidence": [f"{source}/frame_landmark_bounds.json", "tests/master_review/test_kato_bounds.py"],
            "why": "Verified for H1-H100; H=5 session 1 reaches the last observed frame with zero margin but is valid.",
        },
        {
            "id": "S01",
            "claim": "B3 has lower H20 PCA-10 energy than B5-10 in the frozen 175-landmark sample.",
            "status": "SUPPORTED_EXPLORATORY",
            "claim_type": "predictive association",
            "evidence": [f"{source}/statistical_audit.json", f"{source}/independent_model_check.json"],
            "why": "Mean +0.2644, block-5 conditional interval [+0.0750, +0.4031], 64% of landmarks positive; numerical reproduction passes.",
        },
        {
            "id": "S02",
            "claim": "B3 is better in four of five recordings.",
            "status": "SUPPORTED_EXPLORATORY",
            "claim_type": "heterogeneity",
            "evidence": [f"{source}/statistical_audit.json"],
            "why": "Session means +0.300, +0.329, -0.193, +0.407, +0.465; session 2 reverses.",
        },
        {
            "id": "S03",
            "claim": "The B3 advantage generalizes across animals.",
            "status": "UNSUPPORTED",
            "claim_type": "population inference",
            "evidence": [f"{source}/statistical_audit.json"],
            "why": "Recording-level t interval [-0.064, +0.587] includes zero; one-sided sign test p=0.1875; five units.",
        },
        {
            "id": "S04",
            "claim": "B3 improves the conditional mean forecast.",
            "status": "UNSUPPORTED",
            "claim_type": "mean accuracy",
            "evidence": [f"{source}/statistical_audit.json", f"{source}/capacity_audit.json"],
            "why": "PCA-10 MSE +0.0545 with block-5 interval [-0.157, +0.241]; neuron MSE -0.0354 with interval crossing zero.",
        },
        {
            "id": "S05",
            "claim": "The B3 advantage is uniform across horizons and recordings.",
            "status": "INVALIDATED",
            "claim_type": "uniformity",
            "evidence": [f"{source}/statistical_audit.json"],
            "why": "H1 reverses (-0.0637; B5 better), H5/H10/H20 are positive, session 2 reverses.",
        },
        {
            "id": "S06",
            "claim": "The 51% common / 49% residual split is a unique structural partition.",
            "status": "INVALIDATED",
            "claim_type": "decomposition",
            "evidence": [f"{source}/capacity_audit.json"],
            "why": "It is an ordered B4m->B5->B3 decomposition; B1m gives 23.5/76.5 and B0 gives 72.4/27.6.",
        },
        {
            "id": "S07",
            "claim": "The fitted off-diagonal operator identifies stable direct neuron-neuron interactions.",
            "status": "UNSUPPORTED",
            "claim_type": "mechanism",
            "evidence": ["docs/KATO_INTERACTIONS_RESULTS.md", "docs/KATO_INTERACTIONS_IDENTIFIABILITY.md"],
            "why": "Off-diagonal cross-segment correlation 0.052; G3 fails; common-input synthetic confound.",
        },
        {
            "id": "S08",
            "claim": "A named-neuron virtual perturbation predicts a biological intervention.",
            "status": "NOT_EVALUABLE",
            "claim_type": "causal",
            "evidence": ["docs/KATO_FINAL_AUDIT_RESULTS.md"],
            "why": "Model-dependent +1 SD displacement; cross-fit response cosine 0.556; no matched intervention data.",
        },
        {
            "id": "S09",
            "claim": "The ensemble is probabilistically calibrated.",
            "status": "UNSUPPORTED",
            "claim_type": "calibration",
            "evidence": [f"{source}/statistical_audit.json"],
            "why": "B5 95% coverage is 0.725; B3 is 0.938 but calibration was not established in prior calibration audits.",
        },
        {
            "id": "S10",
            "claim": "G4 is a fully prespecified confirmatory PASS.",
            "status": "INVALIDATED",
            "claim_type": "protocol",
            "evidence": ["docs/KATO_FINAL_AUDIT_SIGN_CONVENTION.md", "docs/KATO_FINAL_AUDIT_RESULTS.md"],
            "why": "The sign rule was clarified in the same commit as the results; numerical pass under corrected sign, confirmatory status NOT_ESTABLISHED.",
        },
        {
            "id": "S11",
            "claim": "The residual B3 advantage is information rather than estimator capacity.",
            "status": "NOT_EVALUABLE",
            "claim_type": "capacity",
            "evidence": [f"{source}/capacity_audit.json"],
            "why": "B3 has 7.96-9.95x more parameters and arbitrary residual coordinates; no capacity-matched restricted model was prespecified.",
        },
        {
            "id": "S12",
            "claim": "B5 undercoverage is a sufficient explanation of the energy advantage.",
            "status": "UNSUPPORTED",
            "claim_type": "alternative explanation",
            "evidence": [f"{source}/statistical_audit.json"],
            "why": "Coverage 0.725 vs 0.938 and negligible MSE effect make it plausible, but no decomposition of energy into calibration and resolution establishes sufficiency.",
        },
        {
            "id": "S13",
            "claim": "The Kato test tails were untouched before the final audit.",
            "status": "INVALIDATED",
            "claim_type": "protocol",
            "evidence": ["docs/KATO_FINAL_AUDIT_SIGN_CONVENTION.md", "docs/KATO_FINAL_AUDIT_RESULTS.md"],
            "why": "Earlier phases inspected the same test block; the final audit is an exploratory reanalysis.",
        },
        {
            "id": "S14",
            "claim": "Test frames are touched once, only for final metrics.",
            "status": "INVALIDATED",
            "claim_type": "protocol",
            "evidence": ["docs/METHODOLOGY.md", "src/neurobranch/kato_common_state/models.py:338-396"],
            "why": "B3/B5 are refit at every landmark on all transitions before t, including validation and earlier test frames. No future leakage, but the estimand is online-refit forecasting.",
        },
        {
            "id": "S15",
            "claim": "The reported 95% coverage is joint 10-D calibration.",
            "status": "INVALIDATED",
            "claim_type": "calibration",
            "evidence": ["src/neurobranch/probabilistic/metrics.py:109-120"],
            "why": "coverage_95 is the average of marginal per-dimension interval coverage, not joint ellipsoid coverage.",
        },
        {
            "id": "S16",
            "claim": "The result is ready for confirmatory publication or biological interpretation.",
            "status": "UNSUPPORTED",
            "claim_type": "readiness",
            "evidence": [f"{source}/quality_gate.json"],
            "why": "Exploratory, one reused cohort, animal-level interval includes zero, calibration/capacity confounds, no intervention.",
        },
    ]
    return {
        "schema": "neurobranch.master_review.claims.v1",
        "tier_definitions": {
            "SUPPORTED_EXPLORATORY": "Numerically supported in this dataset; not confirmatory and not generalized.",
            "LIMITED": "Supported only under stated constraints, or a descriptive/archive fact bounded to this material.",
            "UNSUPPORTED": "The available evidence does not establish the claim.",
            "INVALIDATED": "The claim is contradicted by the audit evidence.",
            "NOT_EVALUABLE": "The required data or design is absent.",
        },
        "no_confirmed_tier": True,
        "effect_size_context": effect,
        "claims": claims,
    }


def _corrections() -> dict[str, object]:
    items = [
        ("X01", "DATA_PROVENANCE.md size 128,511,040 bytes", "56,390,975 bytes", "Actual file and OSF API agree; SHA256 still matches.", "HIGH", "Documentation"),
        ("X02", "Five recordings are 'repeated sessions, not five animals'", "Five animals inferred from the publication's n=5 and five distinct acquisition IDs; OSF has no animal key.", "Distinct dates/IDs/trace panels; secondary papers describe five worms. They are not an independent cohort.", "HIGH", "Interpretation"),
        ("X03", "'175 test landmarks (35 per session)'", "35, 35, 34, 38, 33; total 175.", "Computed from per-recording test masks.", "MEDIUM", "Documentation"),
        ("X04", "Common test boundary 2689:3137", "Session 0 only; code uses per-recording test starts.", "Exact split bounds; no out-of-bounds target.", "MEDIUM", "Documentation"),
        ("X05", "G4 fully prespecified confirmatory PASS", "Numerical pass under corrected sign; confirmatory NOT_ESTABLISHED.", "Protocol added with results; sign clarification in same commit.", "HIGH", "Claims"),
        ("X06", "51% common / 49% residual as a unique partition", "Ordered B4m->B5->B3 decomposition; 23.5/76.5 vs B1m and 72.4/27.6 vs B0.", "Arithmetic recomputed.", "MEDIUM", "Interpretation"),
        ("X07", "B3 advantage as a general forecast improvement", "H20 PCA-10 energy only; H1 reverses and MSE intervals cross zero.", "Horizon and representation tables.", "HIGH", "Claims"),
        ("X08", "175 landmarks treated as independent replicates", "Overlapping windows; ~25 non-overlapping H100 proxies; inflation ~7x.", "Landmark stride and horizon.", "HIGH", "Statistics"),
        ("X09", "Test tails untouched", "Reused across earlier phases; final audit exploratory.", "Final audit's own scope statement.", "HIGH", "Protocol"),
        ("X10", "Probabilistic calibration established", "Not established; B5 95% coverage 0.725, B3 0.938.", "Per-landmark coverage.", "HIGH", "Claims"),
        ("X11", "Block-20 interval [+0.092, +0.294] as robust precision", "Conditional within-session interval; recording-resampling interval is [-0.131, +0.440].", "Master-review cluster bootstrap.", "HIGH", "Statistics"),
        ("X12", "B3 residual advantage isolates cross-neuron information", "Not identifiable from capacity; B3 has 7.96-9.95x parameters.", "Parameter counts and B5 residual restriction.", "HIGH", "Mechanism"),
        ("X13", "stateParams rows align with trace rows", "They do not: traces/stateParams rows are 109/109, 135/107, 131/131, 125/126, 129/129; stateParams is unused by the model.", "HDF5 shape audit.", "LOW", "Metadata"),
        ("X14", "DATA_PROVENANCE field names `IDs`, `timeVectorSeconds`", "HDF5 fields are `NeuronNames`, `tv`; the readme uses conceptual MATLAB names.", "Direct HDF5 field listing.", "LOW", "Documentation"),
        ("X15", "Five recordings are an independent cohort", "They are five animals from one laboratory/preparation; no independent cohort.", "Acquisition IDs and publication context.", "MEDIUM", "Inference"),
        ("X16", "Test frames are touched once, only for final metrics", "B3/B5 are refit at every landmark on all observed frames before t, including validation and earlier test frames. No future leakage, but online-refit estimand.", "Rolling fit starts in `kato_common_state/models.py:338-396`.", "HIGH", "Protocol"),
        ("X17", "95% coverage means joint 10-D calibration", "It is the average of marginal per-dimension interval coverage.", "`coverage_summary` calls `interval_coverage` per dimension.", "MEDIUM", "Statistics"),
    ]
    return {
        "schema": "neurobranch.master_review.corrections.v1",
        "items": [
            {
                "id": i,
                "was": was,
                "correct": correct,
                "evidence": evidence,
                "severity": severity,
                "category": category,
                "action": "Documented in master review; frozen historical artifact not rewritten.",
            }
            for i, was, correct, evidence, severity, category in items
        ],
    }


def _code_audit() -> dict[str, object]:
    return {
        "schema": "neurobranch.master_review.code_audit.v1",
        "status": "ARITHMETIC_PASS__INFERENTIAL_CAVEATS",
        "equations": [
            {
                "name": "B3 ridge VAR",
                "source": "src/neurobranch/kato_interactions/models.py:91",
                "equation": "x_{t+1} = b + A x_t + eps_t; ridge 1.0 on standardized features; spectral cap 0.995",
                "verified": True,
            },
            {
                "name": "B5 common-factor AR",
                "source": "src/neurobranch/kato_common_state/models.py:224",
                "equation": "x_{t+1,j} = b_j + a_j x_{t,j} + beta_j^T f_t + eps; f_{t+1}=A_f f_t+b_f",
                "verified": True,
            },
            {
                "name": "energy score",
                "source": "src/neurobranch/probabilistic/metrics.py:24",
                "equation": "E||X-y|| - 0.5 E||X-X'||, distinct pairs",
                "verified": True,
            },
            {
                "name": "moving-block interval",
                "source": "src/neurobranch/kato_interactions/metrics.py:36",
                "equation": "resample contiguous blocks within each session; equal-session pooling; no between-session resampling",
                "verified": True,
            },
        ],
        "findings": [
            {
                "id": "C01",
                "severity": "HIGH",
                "location": "src/neurobranch/kato_interactions/metrics.py:36-76",
                "finding": "The primary interval conditions on the five recordings; it does not resample recordings. A recording-level bootstrap gives a wider interval that includes zero.",
                "impact": "The reported interval is not an animal-level generalization interval.",
            },
            {
                "id": "C02",
                "severity": "HIGH",
                "location": "src/neurobranch/kato_common_state/models.py:224-292",
                "finding": "B5 uses a globally fixed 10-D PCA basis and a diagonal-plus-rank-10 residual operator; B3 uses a local 500-frame full operator with arbitrary residual rows.",
                "impact": "The contrast confounds residual information with adaptivity and capacity.",
            },
            {
                "id": "C03",
                "severity": "HIGH",
                "location": "reports/simulator/kato_common_state/model_per_landmark.csv",
                "finding": "The B5 energy deficit coincides with 95% coverage 0.725 versus B3 0.938, while the MSE contrast is negligible.",
                "impact": "Energy advantage is partly a spread/calibration contrast, not a mean-accuracy result.",
            },
            {
                "id": "C04",
                "severity": "MEDIUM_HIGH",
                "location": "src/neurobranch/kato_common_state/models.py:338-396",
                "finding": "At each landmark, B3/B5 are refit on all transitions before t, including validation and earlier test frames.",
                "impact": "No future leakage, but the estimand is online-refit forecasting, not a frozen train-only model; the 'test touched once' statement is false.",
            },
            {
                "id": "C05",
                "severity": "MEDIUM",
                "location": "src/neurobranch/kato_common_state/models.py:399",
                "finding": "B6 is a coefficient ablation with inherited coefficients, not a refitted restricted model.",
                "impact": "B6 is diagnostic only; B4m/B5 are the clean restrictions.",
            },
            {
                "id": "C06",
                "severity": "LOW",
                "location": "src/neurobranch/data/kato.py:184-240",
                "finding": "The loader preserves numeric trace labels as predictors; these are not cell identities and cannot be aligned across recordings.",
                "impact": "All analyses are within-recording; no cross-animal neuron alignment.",
            },
            {
                "id": "C07",
                "severity": "LOW",
                "location": "data/raw/kato2015/WT_NoStim.mat",
                "finding": "stateParams rows do not match trace rows in sessions 1 and 3; the field is unused.",
                "impact": "No effect on the model, but archive tables are not row-aligned.",
            },
            {
                "id": "C08",
                "severity": "MEDIUM",
                "location": "src/neurobranch/probabilistic/metrics.py:109-120",
                "finding": "coverage_95 is the average of marginal per-dimension interval coverage, not joint 10-D coverage.",
                "impact": "Both models are less calibrated than the headline marginal number suggests.",
            },
        ],
        "independent_check": {
            "status": "PASS",
            "max_abs_error": 7.46e-07,
            "cases": 3,
            "source": "reports/master_review/independent_model_check.json",
        },
    }


def _quality_gate() -> dict[str, object]:
    return {
        "schema": "neurobranch.master_review.quality_gate.v1",
        "overall": "READY_FOR_METHODOLOGICAL_SUPERVISOR_REVIEW; NOT_READY_FOR_CONFIRMATORY_OR_BIOLOGICAL_CLAIMS",
        "gates": [
            {"id": "G-DATA", "question": "Archive identity and hash", "status": "PASS_WITH_CAVEAT", "note": "Hash matches OSF; documented size wrong."},
            {"id": "G-SPLIT", "question": "Per-recording split and purge", "status": "PASS", "note": "No overlap; exact bounds."},
            {"id": "G-LANDMARK", "question": "Forecast targets inside recording/test mask", "status": "PASS", "note": "H1-H100 verified; H5 session 1 reaches last frame with zero margin."},
            {"id": "G-NUMERIC", "question": "Independent arithmetic reproduction", "status": "PASS", "note": "Three landmarks, max abs error 7.46e-7."},
            {"id": "G-DOCS", "question": "Documentation consistency", "status": "FAIL", "note": "Size, 35/session, shared boundary, five-sessions wording, G4 label, test-touched-once, joint-coverage wording."},
            {"id": "G-INFERENCE", "question": "Animal-level generalization", "status": "FAIL", "note": "Recording bootstrap and sign test do not exclude zero."},
            {"id": "G-CALIBRATION", "question": "Probabilistic calibration", "status": "FAIL", "note": "B5 95% coverage 0.725; B3 0.938; not established."},
            {"id": "G-CAPACITY", "question": "Capacity-matched residual information", "status": "NOT_EVALUABLE", "note": "7.96-9.95x parameter ratio; no capacity-matched model."},
            {"id": "G-BIO", "question": "Biological intervention validity", "status": "NOT_EVALUABLE", "note": "No matched perturbation data."},
            {"id": "G-CONFIRM", "question": "Confirmatory status", "status": "NOT_ESTABLISHED", "note": "Sign rule clarified with results; test reused."},
        ],
    }


def _history() -> list[dict[str, str]]:
    return [
        {"phase": "P0", "dates": "2026-10-22/23", "anchor": "f64e761", "question": "Can a reproducible C. elegans platform be built?", "result": "Platform, 105-run blocked suite, 35-run forward stress, NC-MCM and BunDLe artifact checks.", "status": "FOUNDATION", "lesson": "Passing tests and artifact reproduction are not scientific confirmation."},
        {"phase": "P1", "dates": "2026-10-23", "anchor": "5b32c9c..", "question": "Are inherited branching/intervention claims sound?", "result": "Branching p=1.0 reinterpreted; intervention specificity and BunDLe leakage caveats documented.", "status": "AUDIT", "lesson": "Label permutation tests need dependence-preserving nulls."},
        {"phase": "P2", "dates": "2026-10-23", "anchor": "research/phase2-intervention-validation", "question": "Do named-neuron interventions validate a target?", "result": "No validated target; winner's curse and control design exposed.", "status": "NEGATIVE", "lesson": "Model sensitivity is not biological effect."},
        {"phase": "P3", "dates": "2026-10-23", "anchor": "research/phase3-neural-generalization", "question": "Does neural activity add prospective behavioural information in an independent dataset?", "result": "Baskoylu 27 animals: neural increment inconclusive; PCA secondary result not robust.", "status": "NEGATIVE/INCONCLUSIVE", "lesson": "Animal-disjoint evaluation and proper scores matter."},
        {"phase": "P4", "dates": "2026-10-24", "anchor": "research/phase4-turn-direction-generalization", "question": "Does population activity predict turn direction?", "result": "Kramer 56 animals: ΔAUROC +0.0123, CI crosses zero; replication criterion not met.", "status": "NULL", "lesson": "A point estimate is not an increment without uncertainty."},
        {"phase": "P5", "dates": "2026-10-24/25", "anchor": "research/phase5-saav-validation", "question": "Can SAAV be reconstructed and validated?", "result": "Exact reproduction BLOCKED_MISSING_INPUTS; partial reconstruction null.", "status": "BLOCKED/PARTIAL", "lesson": "Missing inputs are a hard boundary, not a tuning problem."},
        {"phase": "P6", "dates": "2026-10-31", "anchor": "research/phase6-causal-atlas", "question": "Do autonomous simulators reproduce dynamics and branching?", "result": "G2/G3/G4 fail; G5 interface pass; G7 not evaluable.", "status": "NEGATIVE", "lesson": "Free-running skill is the right simulator test; one-step R2 is not."},
        {"phase": "P7", "dates": "2026-11-01", "anchor": "phase7", "question": "Can active probing improve target ranking?", "result": "No benchmark superiority; normalised scores withdrawn.", "status": "WITHDRAWN", "lesson": "A truncated denominator is not a ceiling."},
        {"phase": "P8", "dates": "2026-11-01", "anchor": "phase8", "question": "Does the adaptive twin beat the pair baseline?", "result": "Pair gate underpowered (MDE 0.0587 vs SESOI 0.020).", "status": "UNDERPOWERED", "lesson": "A CI excluding zero can still be underpowered."},
        {"phase": "P9", "dates": "2026-11-01", "anchor": "phase9-bounded-benchmark", "question": "What is the measurement ceiling?", "result": "Reliability estimator replaced; normalised-score and ceiling claims corrected.", "status": "CORRECTED", "lesson": "Estimator choice changes the ceiling by ~3x."},
        {"phase": "P10", "dates": "2026-10-29", "anchor": "phase10", "question": "Does a metric-aligned ceiling and coverage study support a benchmark?", "result": "Coverage study negative; pooled ceiling descriptive only.", "status": "NEGATIVE", "lesson": "Coverage-aware ceilings can lower, not raise, claims."},
        {"phase": "P11", "dates": "2026-10-29", "anchor": "phase11", "question": "Can a 0.560 pooled ceiling be published?", "result": "Withdrawn: only 39/126 targets have >=22 animals; extrapolation fails.", "status": "WITHDRAWN", "lesson": "Do not extrapolate a reliability bound outside support."},
        {"phase": "P12", "dates": "2026-10-30", "anchor": "phase12/12R", "question": "Can a hierarchical model calibrate the reliability interval?", "result": "SBC coverage 0.923, KS p=1.4e-6; interval not calibrated.", "status": "FAILED", "lesson": "A point estimate matching is not interval calibration."},
        {"phase": "P13", "dates": "2026-10-31", "anchor": "phase13", "question": "Is the reliability result stable under measurement variants?", "result": "R_obs 0.2606 [0.178, 0.343]; extrapolation and ranking fail.", "status": "DESCRIPTIVE", "lesson": "Report the estimand and its population."},
        {"phase": "P14", "dates": "2026-10-31", "anchor": "phase14", "question": "Does a ZINB measurement model rescue calibration?", "result": "Rejected: 8/12 gates fail; prior-dominated.", "status": "REJECTED", "lesson": "A more complex model is not a calibration fix."},
        {"phase": "P15", "dates": "2026-11-01", "anchor": "phase15", "question": "Which reliability estimand is defensible?", "result": "R_obs 0.2606; R_22 and R_id are different estimands; no ranking.", "status": "BOUNDED", "lesson": "Do not promote a descriptive estimate to a guarantee."},
        {"phase": "K1", "dates": "2026-11-02", "anchor": "23b0ed1", "question": "Is there predictive cross-neuron information in Kato?", "result": "B3 beats B1 on energy (+0.371); operator unstable (r=0.052); structural G7 fails.", "status": "EXPLORATORY", "lesson": "Prediction does not identify coupling."},
        {"phase": "K2", "dates": "2026-11-02/03", "anchor": "358107a", "question": "Does the advantage survive own-history and ten common factors?", "result": "B5-B3 +0.2644; 4/5 sessions; H1 reverses; G4 numerical pass.", "status": "EXPLORATORY", "lesson": "Common factors explain about half under one ordering; capacity unresolved."},
        {"phase": "K3", "dates": "2026-11-03", "anchor": "9f4f833..96a8dd3", "question": "Does the final audit reproduce end-to-end?", "result": "Clean rerun passes; sign ambiguity means G4 confirmatory NOT_ESTABLISHED; decision READY_FOR_SUPERVISOR_REVIEW.", "status": "REPRODUCED/EXPLORATORY", "lesson": "Reproduction is not confirmation."},
        {"phase": "M1", "dates": "2026-11-03", "anchor": "master-review", "question": "Are the data identity, bounds, statistics and claims independently defensible?", "result": "Bounds pass; animal-level interval includes zero; calibration/capacity confounds; documentation corrections required.", "status": "METHODOLOGICAL_REVIEW", "lesson": "The numerical result is real; the inferential claim must be narrower."},
    ]


def main() -> None:
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    diff, effect = _effect_sizes()
    capacity = _capacity(identity, diff, effect)
    stats["paired_effect"] = effect
    stats["calibration_confound"] = {
        "b3_95_coverage": effect["coverage_at_nominal_95"]["B3_population"]["95"],
        "b5_95_coverage": effect["coverage_at_nominal_95"]["B5_common_k10"]["95"],
        "b3_mean_mse": 1.706676973971411,
        "b5_mean_mse": 1.7611812839179795,
        "mse_difference": 0.05450430994656875,
        "mse_interval_block5": [-0.15742619068429753, 0.24090408563595608],
        "interpretation": (
            "B5 is severely undercovered at nominal 95%, while its mean MSE is only "
            "3.1% worse. The energy gap can therefore reflect ensemble spread and "
            "calibration rather than better conditional means."
        ),
    }
    stats["capacity"] = capacity
    _write_json("statistical_audit.json", stats)
    _write_json("capacity_audit.json", capacity)
    _write_json("code_audit.json", _code_audit())
    _write_json("claims_ledger.json", _claims(effect))
    _write_json("corrections.json", _corrections())
    _write_json("quality_gate.json", _quality_gate())

    history = _history()
    with (REPORTS / "history_timeline.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    manifest: dict[str, object] = {
        "schema": "neurobranch.master_review.manifest.v1",
        "branch": "research/kato-final-reproducibility-audit",
        "head": "96a8dd333643de0564247e2c01c98a85511dac82",
        "raw_archive_sha256": identity["archive"]["actual_sha256"],
        "raw_archive_size": identity["archive"]["actual_size_bytes"],
        "files": {},
    }
    for path in sorted(REPORTS.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.name] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    _write_json("manifest.json", manifest)
    print(json.dumps({"wrote": len(manifest["files"]) + 1, "capacity_ratio": [capacity["ratio_min"], capacity["ratio_max"]]}, indent=2))


if __name__ == "__main__":
    main()
