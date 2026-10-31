"""Inventory the public external recordings without loading them into Git.

The script records exact file hashes, trace-level label coverage and stimulus
schedules.  It deliberately does not infer animal independence from filenames.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from neurobranch.config import PROJECT_ROOT, sha256_file
from neurobranch.data.external import ExternalRecording, load_kato_stim, load_nichols
from neurobranch.data.kato import load_kato

DEFAULT_KATO_STIM = PROJECT_ROOT / "data/raw/external/kato2015/WT_Stim.mat"
DEFAULT_KATO_NOSTIM = PROJECT_ROOT / "data/raw/kato2015/WT_NoStim.mat"
DEFAULT_NICHOLS = [
    PROJECT_ROOT / "data/raw/external/nichols2017/npr1_prelet.mat",
    PROJECT_ROOT / "data/raw/external/nichols2017/npr1_let.mat",
    PROJECT_ROOT / "data/raw/external/nichols2017/n2_let.mat",
    PROJECT_ROOT / "data/raw/external/nichols2017/n2_prelet.mat",
]


def _set_summary(records: list[ExternalRecording]) -> dict[str, Any]:
    sets = [set(record.named_labels) for record in records]
    common = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    jaccards = []
    for a, b in combinations(sets, 2):
        denom = len(a | b)
        jaccards.append(len(a & b) / denom if denom else 0.0)
    sizes = [len(s) for s in sets]
    return {
        "n_sessions": len(records),
        "common_named_neurons": sorted(common),
        "n_common_named_neurons": len(common),
        "n_union_named_neurons": len(union),
        "per_session_named_label_counts": sizes,
        "per_session_named_label_count_min": min(sizes) if sizes else 0,
        "per_session_named_label_count_max": max(sizes) if sizes else 0,
        "pairwise_jaccard_min": min(jaccards) if jaccards else None,
        "pairwise_jaccard_max": max(jaccards) if jaccards else None,
        "animal_independence": (
            "OSF files identify recording sessions/dates but contain no animal IDs. "
            "Independence and non-overlap with Kato NoStim are therefore not provable "
            "from metadata; they are not assumed."
        ),
    }


def _aggregate(records: list[ExternalRecording], files: list[Path]) -> dict[str, Any]:
    states: list[str] = []
    for record in records:
        states.extend(record.state_names)
    stimulus = records[0].stimulus if records else {}
    file_counts: dict[str, int] = {}
    for record in records:
        name = Path(record.source_file).name
        file_counts[name] = file_counts.get(name, 0) + 1
    return {
        "n_sessions": len(records),
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "n_sessions": file_counts.get(path.name, 0),
            }
            for path in files
        ],
        "n_frames_min": min(r.n_frames for r in records) if records else 0,
        "n_frames_max": max(r.n_frames for r in records) if records else 0,
        "duration_seconds_min": min(r.duration_seconds for r in records) if records else None,
        "duration_seconds_max": max(r.duration_seconds for r in records) if records else None,
        "fps_min": min(r.fps for r in records) if records else None,
        "fps_max": max(r.fps for r in records) if records else None,
        "n_traces_min": min(r.n_traces for r in records) if records else 0,
        "n_traces_max": max(r.n_traces for r in records) if records else 0,
        "n_labeled_traces_min": min(int(r.named_mask.sum()) for r in records) if records else 0,
        "n_labeled_traces_max": max(int(r.named_mask.sum()) for r in records) if records else 0,
        "state_names": sorted(set(states)),
        "stimulus": stimulus,
        **_set_summary(records),
    }


def _kato_nostim_summary(path: Path) -> dict[str, Any]:
    dataset = load_kato(path, verify=True)
    sets = [set(r.neurons) for r in dataset.recordings]
    common = set.intersection(*sets) if sets else set()
    durations = [float(r.time_seconds[-1] - r.time_seconds[0]) for r in dataset.recordings]
    return {
        "n_sessions": len(dataset.recordings),
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": dataset.source_sha256,
            }
        ],
        "n_frames_min": min(r.n_frames for r in dataset.recordings),
        "n_frames_max": max(r.n_frames for r in dataset.recordings),
        "duration_seconds_min": min(durations),
        "duration_seconds_max": max(durations),
        "fps_min": min(r.fps for r in dataset.recordings),
        "fps_max": max(r.fps for r in dataset.recordings),
        "n_traces_min": min(r.n_neurons for r in dataset.recordings),
        "n_traces_max": max(r.n_neurons for r in dataset.recordings),
        "n_common_named_neurons": len(common),
        "common_named_neurons": sorted(common),
        "animal_independence": (
            "Five recording sessions; no animal IDs in the OSF file. "
            "The test block has already been reused in Phase 5/6."
        ),
        "stimulus": {"identity": "O2", "type": "constant hyperoxia", "concentrations_percent": [21]},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kato-stim", type=Path, default=DEFAULT_KATO_STIM)
    parser.add_argument("--kato-nostim", type=Path, default=DEFAULT_KATO_NOSTIM)
    parser.add_argument("--nichols", type=Path, nargs="+", default=DEFAULT_NICHOLS)
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/external_validation")
    args = parser.parse_args(argv)

    stim = load_kato_stim(args.kato_stim, verify=True)
    nichols: list[ExternalRecording] = []
    for path in args.nichols:
        nichols.extend(load_nichols(path, verify=True))

    sessions: list[dict[str, Any]] = []
    for record in [*stim, *nichols]:
        row = record.summary()
        row["file"] = Path(record.source_file).name
        row["stimulus"] = json.dumps(row["stimulus"], sort_keys=True)
        sessions.append(row)

    nichols_summary = _aggregate(nichols, list(args.nichols))
    nostim_summary = _kato_nostim_summary(args.kato_nostim)
    stim_sets = [set(r.named_labels) for r in stim]
    nostim_sets = [set(r.neurons) for r in load_kato(args.kato_nostim, verify=False).recordings]
    cross_common = (
        set.intersection(*stim_sets) & set.intersection(*nostim_sets) if stim_sets and nostim_sets else set()
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "Recording sessions are not proven independent animals. Oxygen changes are measured "
            "environmental input, not targeted neuronal stimulation."
        ),
        "datasets": {
            "kato2015_wt_stim": {
                **_aggregate(stim, [args.kato_stim]),
                "provenance": "https://osf.io/2395t/",
                "paper_doi": "10.1016/j.cell.2015.09.034",
                "preparation": "immobilized whole-brain calcium imaging (dLDS paper description)",
                "behavior": "four coarse fictive states",
                "oxygen_segment": "21% for 0-360 s; 4/21% steps every 30 s from 360 s",
            },
            "kato2015_wt_nostim": {
                **nostim_summary,
                "provenance": "https://osf.io/2395t/",
                "paper_doi": "10.1016/j.cell.2015.09.034",
                "preparation": "immobilized whole-brain calcium imaging",
                "behavior": "eight fine fictive states",
            },
            "nichols2017": {
                **nichols_summary,
                "provenance": "https://osf.io/kbf38/",
                "paper_doi": "10.1126/science.aam6851",
                "preparation": "brainwide calcium imaging during lethargus; OSF metadata does not document immobilization",
                "behavior": "five/six states including quiescence",
                "genotypes": "npr-1 and N2 backgrounds; PreLet and Let stages",
            },
        },
        "cross_dataset": {
            "kato_stim_vs_nostim_common_named_neurons": sorted(cross_common),
            "kato_stim_vs_nostim_n_common_named_neurons": len(cross_common),
            "note": (
                "A common name is not a shared panel: trace order, numeric trace IDs, "
                "coverage and multi-neuron labels differ. No frozen model transfer is licensed."
            ),
        },
        "known_discrepancies": [
            "The dLDS paper's Table 3 trace counts differ from the raw OSF HDF5 shapes for some Kato worms; raw file shapes are authoritative here.",
            "Trace-level labels can name several neurons and names can recur across traces; they are not one-to-one cell identities.",
        ],
    }

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "dataset_inventory.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(sessions).to_csv(args.reports / "dataset_inventory_sessions.csv", index=False)
    print(json.dumps({k: v["n_sessions"] for k, v in payload["datasets"].items()}, indent=2))
    print(f"common Kato Stim/NoStim named neurons: {len(cross_common)}")
    print(f"wrote {args.reports / 'dataset_inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
