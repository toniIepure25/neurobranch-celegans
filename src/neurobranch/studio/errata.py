"""Documentation errata and their active-surface treatment.

The frozen historical artifacts are not rewritten.  This module records where
the current Studio surface stops repeating each correction, and which
scientific limitations remain after the wording is fixed.
"""

from __future__ import annotations

from neurobranch.studio.metadata import StudioSnapshot

RESOLUTIONS: dict[str, dict[str, str]] = {
    "X01": {
        "surface": "Overview · provenance, README, STUDIO_SCIENTIFIC_EVIDENCE.md",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "Archive size and SHA256 are read from the identity audit.",
    },
    "X02": {
        "surface": "Overview · provenance; handoff demo",
        "status": "CORRECTED_WITH_INFERRED_LINKAGE",
        "note": "Five recordings are treated as five inferred animals from one preparation; OSF exposes no animal key.",
    },
    "X03": {
        "surface": "3D dynamics · landmark selector; tests",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "Per-recording counts are 35/35/34/38/33.",
    },
    "X04": {
        "surface": "Activity · split overlay; STUDIO_SCIENTIFIC_EVIDENCE.md",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "Each recording has its own split boundaries.",
    },
    "X05": {
        "surface": "Evidence · quality gate; no confirmatory PASS label",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "The numerical gate passed; confirmatory status remains NOT_ESTABLISHED.",
    },
    "X06": {
        "surface": "Evidence · capacity; STUDIO_SCIENTIFIC_EVIDENCE.md",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "The 51/49 split is an ordered arithmetic decomposition, not a unique partition.",
    },
    "X07": {
        "surface": "Overview; 3D dynamics; models; evidence",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "Only H20 PCA-10 energy is claimed; H1 reverses and MSE is null.",
    },
    "X08": {
        "surface": "Evidence · inference-unit panel",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "The 175 overlapping landmarks are never described as independent replicates.",
    },
    "X09": {
        "surface": "Overview · protocol banner; STUDIO_SCIENTIFIC_EVIDENCE.md",
        "status": "DISCLOSED_AS_EXPLORATORY",
        "note": "The test tail is reused across earlier phases; this is not a fresh confirmatory test.",
    },
    "X10": {
        "surface": "Evidence · calibration tab",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "Calibration is not established; B5 coverage is poor.",
    },
    "X11": {
        "surface": "Evidence · forest plot; all intervals labelled by resampling unit",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "Conditional block-5 and recording-resampling intervals are separate objects.",
    },
    "X12": {
        "surface": "Models · capacity warning; Evidence",
        "status": "DISCLOSED_AS_UNIDENTIFIED",
        "note": "The residual advantage is not identifiable as cross-neuron information from this design.",
    },
    "X13": {
        "surface": "Provenance · data-shape disclosure",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "stateParams rows do not align with all trace rows and are unused by the model.",
    },
    "X14": {
        "surface": "Provenance · HDF5 field names",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "NeuronNames and tv are the actual fields.",
    },
    "X15": {
        "surface": "Overview; Studio footer",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "One laboratory/preparation; no independent cohort.",
    },
    "X16": {
        "surface": "Models · refit protocol; STUDIO_REPRODUCIBILITY.md",
        "status": "DISCLOSED_AS_ONLINE_REFIT",
        "note": "No future leakage, but every landmark refits on observed past including earlier test frames; the estimand is an online-refit one.",
    },
    "X17": {
        "surface": "Evidence · calibration",
        "status": "CORRECTED_IN_ACTIVE_SURFACE",
        "note": "Coverage is marginal per dimension, not joint 10-D calibration.",
    },
}


def errata_rows(snapshot: StudioSnapshot) -> list[dict[str, str]]:
    rows = []
    for item in snapshot.corrections:
        resolution = RESOLUTIONS[item["id"]]
        rows.append(
            {
                "id": item["id"],
                "severity": item["severity"],
                "category": item["category"],
                "was": item["was"],
                "correct": item["correct"],
                "active_surface": resolution["surface"],
                "status": resolution["status"],
                "note": resolution["note"],
            }
        )
    return rows


def counts(rows: list[dict[str, str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        result[row["status"]] = result.get(row["status"], 0) + 1
    return dict(sorted(result.items()))
