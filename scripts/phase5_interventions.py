"""Phase 5E: intervention evidence map.

No trial-level optogenetic observations are present in the public Dryad archive.
This script therefore records published group-level findings, available
observations, simulator predictions (none run) and hypotheses separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("reports/phase5/interventions")
PAPER = "Kramer et al., Nat Neurosci 29:1408-1424 (2026), doi:10.1038/s41593-026-02257-5"
DRYAD = "Dryad doi:10.5061/dryad.8sf7m0cz2 (version 5, 82 files)"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "claim_id": "P1",
            "target": "RIM",
            "manipulation": "optogenetic activation",
            "evidence_type": "published_group_level",
            "source": PAPER,
            "observation_level": "group/recording summary in paper",
            "available_in_public_archive": "no trial-level stimulation table in Dryad",
            "trial_level_available": False,
            "causal_interpretation": "manipulation_supported_only_as_reported",
            "what_is_supported": "The paper reports that optogenetic RIM manipulation changes reversal parameters.",
            "what_is_not_supported": "It does not establish SAAV-specific control or a turn-direction decoder.",
            "status": "PUBLISHED_FINDING_NOT_REANALYSED",
        },
        {
            "claim_id": "P2",
            "target": "RIM",
            "manipulation": "optogenetic inhibition",
            "evidence_type": "published_group_level",
            "source": PAPER,
            "observation_level": "group/recording summary in paper",
            "available_in_public_archive": "no trial-level stimulation table in Dryad",
            "trial_level_available": False,
            "causal_interpretation": "manipulation_supported_only_as_reported",
            "what_is_supported": "The paper reports an effect of RIM inhibition on reversal speed/length.",
            "what_is_not_supported": "The archive cannot be used to reproduce cell-specific trial-level effects.",
            "status": "PUBLISHED_FINDING_NOT_REANALYSED",
        },
        {
            "claim_id": "P3",
            "target": "AIB",
            "manipulation": "genetic silencing (inx-1::unc-103[gof])",
            "evidence_type": "published_group_level",
            "source": PAPER,
            "observation_level": "group/recording summary in paper",
            "available_in_public_archive": "no trial-level silencing table in Dryad",
            "trial_level_available": False,
            "causal_interpretation": "genetic_manipulation_reported",
            "what_is_supported": "The paper reports changes in reversal/turn statistics after AIB silencing.",
            "what_is_not_supported": "This is not an acute SAAV stimulation and does not validate our decoder.",
            "status": "PUBLISHED_FINDING_NOT_REANALYSED",
        },
        {
            "claim_id": "P4",
            "target": "tdc-1 / tyramine signalling",
            "manipulation": "mutant background (n3419/n3420/n3421)",
            "evidence_type": "published_group_level",
            "source": PAPER,
            "observation_level": "genotype-level distributions in paper and tdc_enc.csv",
            "available_in_public_archive": "tdc1.wbData.mat and tdc_enc.csv exist; no acute trial table",
            "trial_level_available": False,
            "causal_interpretation": "genotype_shift_not_acute_cell_specific",
            "what_is_supported": "tdc-1 changes behaviour and activity distributions.",
            "what_is_not_supported": "WT versus tdc-1 is not an acute cell-specific stimulation; it cannot isolate SAAV causality.",
            "status": "PARTIAL_PUBLIC_DATA_OBSERVATIONAL",
        },
        {
            "claim_id": "P5",
            "target": "SAAV",
            "manipulation": "none in public archive",
            "evidence_type": "hypothesis",
            "source": "Phase 5 reasoning; paper proposes a role but public data lack cell-specific perturbation",
            "observation_level": "not observed",
            "available_in_public_archive": "SAAV calcium traces only; no SAAV-specific stimulation",
            "trial_level_available": False,
            "causal_interpretation": "not_tested",
            "what_is_supported": "SAAV activity is correlated with turn-related variables in the published data.",
            "what_is_not_supported": "No causal necessity or sufficiency claim is supported by the public archive or this project.",
            "status": "UNTESTED_CAUSAL_HYPOTHESIS",
        },
        {
            "claim_id": "P6",
            "target": "compact head-steering circuit (SAAV, SAAD, SMD, RMD, RIV, SMB)",
            "manipulation": "none in public archive",
            "evidence_type": "observational_decoding",
            "source": "Phase 5 mechanism analysis (this repository)",
            "observation_level": "single-event calcium windows; animal-disjoint decoding",
            "available_in_public_archive": "yes for observational traces",
            "trial_level_available": False,
            "causal_interpretation": "encoding_only",
            "what_is_supported": "Incremental decoding and cross-animal stability can be estimated observationally.",
            "what_is_not_supported": "Encoding/decoding is not causation; ablation results are model ablations, not biological ones.",
            "status": "OBSERVATIONAL_ONLY",
        },
        {
            "claim_id": "P7",
            "target": "any candidate neuron",
            "manipulation": "virtual perturbation / optimisation",
            "evidence_type": "simulator_prediction",
            "source": "Phase 5 decision",
            "observation_level": "none",
            "available_in_public_archive": "not applicable",
            "trial_level_available": False,
            "causal_interpretation": "not_run",
            "what_is_supported": "Nothing; no model passed the validation gate for virtual intervention optimisation.",
            "what_is_not_supported": "Do not interpret any simulated effect as a biological intervention.",
            "status": "NOT_RUN_NO_VALIDATED_MODEL",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "intervention_evidence.csv", index=False)
    summary = {
        "status": "INTERVENTION_EVIDENCE_MAP",
        "source_paper": PAPER,
        "source_data": DRYAD,
        "trial_level_optogenetics_available": False,
        "pseudo_trials_created": False,
        "virtual_interventions_run": False,
        "n_published_findings": int((frame["evidence_type"] == "published_group_level").sum()),
        "n_hypotheses": int((frame["evidence_type"] == "hypothesis").sum()),
        "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
    }
    (OUT / "intervention_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
