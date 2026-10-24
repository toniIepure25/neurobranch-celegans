"""Phase 4 dataset audit and provenance report.

Reads the raw Kramer et al. archive, validates every recording against its HDF5
raw-trace file, and writes the inventory, neuron coverage, behaviour-event
inventory and the inferred D/V marker sanity check.  No outcome model is fit.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.behavior.prospective import causal_zscore
from neurobranch.data.kramer import POST_TURN_FRAMES, load_all

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "phase4"
REPORT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "raw" / "kramer2026"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    recs, audit = load_all()
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    by_path = {f["path"]: f for f in manifest["files"]}
    inventory = []
    for rec in recs:
        raw_path = rec.quality.get("h5_path", "")
        inv = {
            "animal_id": rec.animal_id,
            "cohort": rec.cohort,
            "source_mat": rec.quality.get("source_mat", ""),
            "row_index": rec.quality.get("row_index", -1),
            "h5_file": Path(raw_path).name if raw_path else "",
            "n_frames": rec.n_frames,
            "dt_s": rec.dt,
            "duration_s": rec.n_frames * rec.dt,
            "n_neurons": rec.n_neurons,
            "n_labelled_traces": rec.n_labelled,
            "orientation_sign": rec.quality.get("orientation_sign", 0),
            "reversal_count": len(rec.reversal_onsets),
            "odor_frames": int(np.sum(rec.odor_idx > 0)),
            "odor_present": bool(np.any(rec.odor_idx > 0)),
            "h5_sha256": by_path.get(Path(raw_path).name, {}).get("sha256", "") if raw_path else "",
            "h5_size": by_path.get(Path(raw_path).name, {}).get("size", 0) if raw_path else 0,
            "trace_source": rec.quality.get("trace_source", ""),
        }
        inventory.append(inv)
    inv_df = pd.DataFrame(inventory).sort_values(["cohort", "animal_id"])
    inv_df.to_csv(REPORT / "dataset_inventory.csv", index=False)

    # Cohort-level summary.
    cohort = []
    for name, g in inv_df.groupby("cohort", sort=True):
        cohort.append(
            {
                "cohort": name,
                "recordings": len(g),
                "frames": int(g["n_frames"].sum()),
                "hours": float(g["n_frames"].sum() * 0.6 / 3600),
                "neurons_median": float(g["n_neurons"].median()),
                "labelled_traces": int(g["n_labelled_traces"].sum()),
                "reversals": int(g["reversal_count"].sum()),
                "dt_median_s": float(g["dt_s"].median()),
            }
        )
    pd.DataFrame(cohort).to_csv(REPORT / "cohort_inventory.csv", index=False)

    # Neuron coverage by label.
    long_events = pd.concat(
        [r.event_table(min_length=12) for r in recs], ignore_index=True
    )
    long_events = long_events[long_events["valid_post"]]
    per_label: dict[str, dict[str, object]] = {}
    for rec in recs:
        for label in sorted(set(rec.identified_indices)):
            d = per_label.setdefault(
                label,
                {
                    "label": label,
                    "n_animals": 0,
                    "n_wt": 0,
                    "n_wt_octanol": 0,
                    "n_wt_diacetyl": 0,
                    "n_tdc1_octanol": 0,
                    "n_events_with_label": 0,
                    "n_dorsal_events": 0,
                    "n_ventral_events": 0,
                },
            )
            d["n_animals"] += 1
            if rec.cohort != "tdc1_octanol":
                d["n_wt"] += 1
            d[f"n_{rec.cohort}"] += 1
            ev = long_events[long_events["animal_id"] == rec.animal_id]
            d["n_events_with_label"] += len(ev)
            d["n_dorsal_events"] += int((ev["direction"] == "dorsal").sum())
            d["n_ventral_events"] += int((ev["direction"] == "ventral").sum())
    cov = pd.DataFrame(per_label.values())
    cov["wt_fraction"] = cov["n_wt"] / max(1, int((inv_df["cohort"] != "tdc1_octanol").sum()))
    cov["marker"] = cov["label"].isin(["SAAV", "RMDV", "RMDD", "SMDV", "SMDD", "RIV", "SMBV", "SMBD"])
    cov = cov.sort_values(["marker", "n_wt", "label"], ascending=[False, False, True])
    cov.to_csv(REPORT / "neuron_coverage.csv", index=False)

    # Behaviour/event inventory per recording.
    event_rows = []
    for rec in recs:
        ev = rec.event_table(min_length=0)
        ev12 = ev[ev["duration_frames"] >= 12]
        valid12 = ev12[ev12["valid_post"]]
        pre_speed = []
        for row in ev12.itertuples(index=False):
            pre_speed.append(float(np.mean(np.abs(rec.velocity[row.start : row.end]))))
        event_rows.append(
            {
                "animal_id": rec.animal_id,
                "cohort": rec.cohort,
                "n_reversals": len(ev),
                "n_long_12": len(ev12),
                "n_long_12_valid_post": len(valid12),
                "n_dorsal_12": int((valid12["direction"] == "dorsal").sum()),
                "n_ventral_12": int((valid12["direction"] == "ventral").sum()),
                "n_near_zero_12": int(valid12["near_zero"].sum()),
                "fraction_dorsal_12": float((valid12["direction"] == "dorsal").mean()) if len(valid12) else np.nan,
                "both_classes_12": bool(valid12["direction"].nunique() == 2),
                "mean_reversal_frames": float(ev["duration_frames"].mean()) if len(ev) else np.nan,
                "median_reversal_frames": float(ev["duration_frames"].median()) if len(ev) else np.nan,
                "mean_speed_long_12": float(np.mean(pre_speed)) if pre_speed else np.nan,
                "mean_post_dv_12": float(valid12["post_dv"].mean()) if len(valid12) else np.nan,
                "n_events_with_odor": int(ev["odor_present"].sum()),
            }
        )
    ev_df = pd.DataFrame(event_rows).sort_values(["cohort", "animal_id"])
    ev_df.to_csv(REPORT / "behavior_event_inventory.csv", index=False)

    # Marker sanity: causal z activity just before the offset, grouped by the
    # post-offset D/V label.  This checks the inferred sign convention only; it
    # is not a turn-direction model.
    marker_labels = ["SAAV", "RMDV", "RMDD", "SMDV", "SMDD", "RIV", "SMBV", "SMBD"]
    marker_rows = []
    z_cache: dict[str, np.ndarray] = {}
    event_cache: dict[str, pd.DataFrame] = {}
    for rec in recs:
        if rec.identified_indices:
            z_cache[rec.animal_id] = causal_zscore(rec.traces_raw)
            ev = rec.event_table(min_length=12)
            event_cache[rec.animal_id] = ev[ev["valid_post"]]
    for label in marker_labels:
        for cohort_filter in ("wt_diacetyl", "wt_octanol", "all_wt"):
            pre_d, pre_v, post_d, post_v = [], [], [], []
            for rec in recs:
                if cohort_filter == "all_wt" and rec.cohort == "tdc1_octanol":
                    continue
                if cohort_filter != "all_wt" and rec.cohort != cohort_filter:
                    continue
                idx = rec.identified_indices.get(label)
                if idx is None:
                    continue
                z = z_cache[rec.animal_id]
                ev = event_cache[rec.animal_id]
                for row in ev.itertuples(index=False):
                    pre = float(np.mean(z[max(0, row.end - 4) : row.end, idx]))
                    post = float(np.mean(z[row.end : min(rec.n_frames, row.end + POST_TURN_FRAMES), idx]))
                    if row.direction == "dorsal":
                        pre_d.append(pre)
                        post_d.append(post)
                    else:
                        pre_v.append(pre)
                        post_v.append(post)
            marker_rows.append(
                {
                    "label": label,
                    "cohort": cohort_filter,
                    "n_dorsal": len(pre_d),
                    "n_ventral": len(pre_v),
                    "pre_mean_dorsal": float(np.mean(pre_d)) if pre_d else np.nan,
                    "pre_mean_ventral": float(np.mean(pre_v)) if pre_v else np.nan,
                    "pre_diff_dorsal_minus_ventral": float(np.mean(pre_d) - np.mean(pre_v))
                    if pre_d and pre_v
                    else np.nan,
                    "post_mean_dorsal": float(np.mean(post_d)) if post_d else np.nan,
                    "post_mean_ventral": float(np.mean(post_v)) if post_v else np.nan,
                    "post_diff_dorsal_minus_ventral": float(np.mean(post_d) - np.mean(post_v))
                    if post_d and post_v
                    else np.nan,
                }
            )
    pd.DataFrame(marker_rows).to_csv(REPORT / "marker_sanity.csv", index=False)

    # Audit reasons and identity counts.
    reason_counts = Counter(a.get("reason", "") for a in audit)
    summary = {
        "dataset": {
            "doi": manifest["dataset_doi"],
            "id": manifest["dataset_id"],
            "version_id": manifest["version_id"],
            "version_number": manifest["version_number"],
            "license": manifest["license"],
            "files": len(manifest["files"]),
            "bytes": manifest["storage_size"],
        },
        "recordings": len(recs),
        "frames": int(inv_df["n_frames"].sum()),
        "cohorts": {k: int(v) for k, v in inv_df["cohort"].value_counts().items()},
        "events_total": int(len(pd.concat([r.event_table(min_length=0) for r in recs]))),
        "events_long_12_valid": int(len(long_events)),
        "directions_long_12_valid": {
            k: int(v) for k, v in long_events["direction"].value_counts().items()
        },
        "near_zero_long_12": int(long_events["near_zero"].sum()),
        "unique_identified_labels": int(len(cov)),
        "markers_present": {label: int(cov.loc[cov["label"] == label, "n_wt"].iloc[0]) if (cov["label"] == label).any() else 0 for label in marker_labels},
        "audit_reason_counts": dict(reason_counts),
        "source_note": "HDF5 gcamp/trace_array_original is used; MAT gcamp_trace is the whole-recording z-score and is not used as raw.",
        "direction_note": "dv is the orientation-corrected D/V head angle; positive dv = dorsal. Raw head_angle has an arbitrary per-recording camera sign and is not used for labels.",
        "event_note": "Event intervals are 0-based, end-exclusive. Long events require >=12 frames and a valid post-turn mean over 2 frames.",
    }
    (REPORT / "dataset_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Checksums for the core files that the analysis actually reads.
    checksum_rows = []
    for p in sorted(DATA.glob("files/*")):
        if p.suffix.lower() not in {".mat", ".h5", ".csv", ".md", ".xlsx"}:
            continue
        meta = by_path.get(p.name, {})
        digest = sha256(p)
        checksum_rows.append(
            {
                "file": p.name,
                "bytes": p.stat().st_size,
                "sha256": digest,
                "dryad_sha256": meta.get("sha256", ""),
                "match": digest == meta.get("sha256", "") if meta else None,
            }
        )
    pd.DataFrame(checksum_rows).to_csv(REPORT / "data_checksums.csv", index=False)

    print(json.dumps(summary, indent=2))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
