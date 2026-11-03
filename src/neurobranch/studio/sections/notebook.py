"""Research notebook: history, course, errata, reproducibility, and next steps."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from neurobranch.studio.components import (
    badge,
    caption,
    evidence_badge,
    note,
    section_header,
    warning,
)
from neurobranch.studio.course import load_chapters
from neurobranch.studio.errata import counts, errata_rows
from neurobranch.studio.metadata import StudioSnapshot

DOC_INDEX = [
    ("STUDIO_ARCHITECTURE.md", "Application layers, boundaries, and data flow."),
    ("STUDIO_DESIGN_SYSTEM.md", "Palette, typography, components, accessibility, and Plotly conventions."),
    ("STUDIO_3D_SCIENTIFIC_MAPPING.md", "What the axes are, alignment, uncertainty overlays, and non-claims."),
    ("STUDIO_USER_GUIDE.md", "Navigation, 3D controls, export, and a self-guided tour."),
    ("STUDIO_SCIENTIFIC_EVIDENCE.md", "Claim ledger, estimands, intervals, and evidence boundaries."),
    ("STUDIO_DOCUMENTATION_ERRATA.md", "All 17 master-review corrections and their active-surface treatment."),
    ("STUDIO_REPRODUCIBILITY.md", "Commands, artifact hashes, tests, and environment assumptions."),
    ("STUDIO_TESTING_AND_QA.md", "Unit, integration, app-smoke, and visual-QA record."),
    ("STUDIO_SUPERVISOR_DEMO.md", "Eight-minute demonstration script for Denis Kovačević."),
    ("MASTER_REVIEW_EDUCATIONAL_COURSE.md", "Existing 16-chapter graduate course; read in the Course tab."),
]


def render(snapshot: StudioSnapshot, *, light: bool = False) -> None:
    section_header(
        "07 · RESEARCH NOTEBOOK",
        "The history, the course, the corrections, and the next honest experiment.",
        "This section is a navigation layer over existing project evidence. It is not a "
        "second textbook and it does not replace the Master Review artifacts.",
    )
    tabs = st.tabs(["Research timeline", "Master-review course", "Documentation errata", "Reproducibility", "Next experiments"])
    with tabs[0]:
        st.markdown("#### Fourteen phases, including the negative results")
        st.markdown(
            "The project moved through foundation, external replication, simulator "
            "validation, benchmark construction, and a final audit. Preserving the "
            "negative and withdrawn phases is part of the scientific record."
        )
        for _, row in snapshot.history.iterrows():
            status = str(row.get("status", "")).strip()
            if status.lower() == "nan":
                status = "UNRECORDED"
            st.markdown(
                f"""
                <div class="nb-panel">
                  <div style="display:flex;justify-content:space-between;gap:1rem;align-items:baseline">
                    <div><span class="nb-kicker">{row["phase"]}</span>
                    <b style="font-size:0.95rem">{row["question"]}</b></div>
                    <div>{badge(status, "danger" if status in {"NEGATIVE", "WITHDRAWN", "BLOCKED/PARTIAL", "NEGATIVE/INCONCLUSIVE"} else "control")}</div>
                  </div>
                  <div style="font-size:0.84rem;line-height:1.55;margin-top:0.45rem">
                    <div><b>Result.</b> {row["result"]}</div>
                    <div style="color:inherit;opacity:0.7;margin-top:0.25rem"><b>Lesson.</b> {row["lesson"]}</div>
                    <div class="nb-mono" style="margin-top:0.35rem">{row["dates"]} · {row["anchor"]}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption(
            "Historical phases may use other datasets. Their numbers are not pooled with "
            "the Kato WT_NoStim analysis shown in this Studio. See "
            "docs/MASTER_REVIEW_RESEARCH_HISTORY.md."
        )

    with tabs[1]:
        chapters = load_chapters()
        st.markdown(
            "The existing 16-chapter course is rendered directly from "
            "`docs/MASTER_REVIEW_EDUCATIONAL_COURSE.md`. Each chapter asks for an "
            "explanation of the inferential object, not a citation of a test result."
        )
        choice = st.selectbox(
            "Chapter",
            list(range(len(chapters))),
            format_func=lambda i: f"{chapters[i].number}. {chapters[i].title}",
        )
        chapter = chapters[choice]
        st.markdown(f"### Chapter {chapter.number} — {chapter.title}")
        st.markdown(chapter.body)
        st.caption(
            "The chapter text is read from the authoritative course file. Links inside "
            "the chapter are repository-relative and are intended for an editor or "
            "rendered documentation site."
        )

    with tabs[2]:
        rows = errata_rows(snapshot)
        st.markdown(
            "All 17 corrections are listed with the original wording, the corrected "
            "wording, the active surface that stops repeating the error, and the "
            "scientific limitation that remains. Historical artifacts are preserved; "
            "they were not rewritten."
        )
        c = counts(rows)
        st.markdown(
            " ".join(badge(f"{k} {v}", "control") for k, v in c.items()),
            unsafe_allow_html=True,
        )
        frame = pd.DataFrame(rows)
        st.dataframe(
            frame[["id", "severity", "status", "was", "correct", "active_surface", "note"]],
            use_container_width=True,
            hide_index=True,
            height=460,
        )
        with st.expander("Read one correction in full", expanded=False):
            chosen = st.selectbox("Correction", frame["id"].tolist())
            row = frame[frame["id"] == chosen].iloc[0]
            st.markdown(f"**Was.** {row['was']}")
            st.markdown(f"**Correct.** {row['correct']}")
            st.markdown(f"**Active surface.** {row['active_surface']}")
            st.markdown(f"**Status.** `{row['status']}`")
            st.markdown(f"**Remaining limitation.** {row['note']}")
        warning(
            "Correcting a sentence does not correct the underlying science. In "
            "particular: the confirmatory status is NOT_ESTABLISHED (X05); the test "
            "tail was reused (X09); the residual mechanism is unidentified (X12); and "
            "the model is refit on an online past that includes earlier test frames "
            "(X16)."
        )

    with tabs[3]:
        st.markdown("#### Source-of-truth hierarchy")
        st.markdown(
            "1. Raw archive identity and hash: `dataset_identity_audit.json`.\n"
            "2. Paired scores: `reports/simulator/kato_common_state/model_per_landmark.csv`.\n"
            "3. Intervals and inference: `statistical_audit.json`, `uncertainty_summary.json`, `cluster_bootstrap.csv`.\n"
            "4. Claims wording: `claims_ledger.json`.\n"
            "5. Corrections: `corrections.json`.\n"
            "6. Studio display: this application; it does not create new inferential results."
        )
        st.markdown("#### Reproduce")
        st.code(
            "python -X utf8 scripts/studio_build_trajectories.py\n"
            "python -X utf8 -m pytest -q\n"
            "python -X utf8 -m ruff check src/neurobranch/studio app/neurobranch_studio.py\n"
            "python -X utf8 -m streamlit run app/neurobranch_studio.py",
            language="powershell",
        )
        st.markdown(
            "The trajectory builder refits B3/B5 at every landmark with H=100 innovations "
            "and stores the first 20 steps. Its score checks are a storage-path check, "
            "not a new inferential test."
        )
        st.markdown("#### Studio documents")
        for name, description in DOC_INDEX:
            st.markdown(f"- `docs/{name}` — {description}")
        manifest = snapshot.manifest
        st.markdown("#### Historical Master Review manifest")
        st.code(
            f"branch  {manifest.get('branch')}\n"
            f"head    {manifest.get('head')}\n"
            f"sha256  {manifest.get('raw_archive_sha256')}\n"
            f"size    {manifest.get('raw_archive_size')} bytes",
            language="text",
        )
        st.caption(
            "This is the historical audit manifest, not the current Git HEAD. The Studio "
            "manifest with artifact hashes is reports/scientific_studio/manifest.json."
        )

    with tabs[4]:
        st.markdown("#### What should happen next")
        st.markdown(
            "The audit stops at the observational boundary. The next step is not another "
            "architecture or hyperparameter search on the reused split."
        )
        items = [
            ("Independent-animal replication", "Pre-register the H20 endpoint, baseline family, and cross-neuron contrast; the unit is the animal, not the landmark."),
            ("Matched neuron-specific perturbation", "Identified target, sham/control, dose, time window, held-out animals and targets, pre-registered direction."),
            ("Measure or manipulate the common input", "Test whether cross-neuron coefficients survive conditioning on an independent common drive."),
            ("Pre-registered operator-stability test", "Set the minimum cross-segment coefficient correlation and mode agreement before fitting."),
            ("Independent behavioural readout", "Replace derived labels as the only readout; pre-specify state resolution."),
            ("Perturbation-support stress test", "Use a stronger density model than the marginal percentile envelope and report the OOD fraction."),
        ]
        for i, (title, body) in enumerate(items, start=1):
            st.markdown(f"**{i}. {title}.** {body}")
        st.markdown(
            "Proceed to an intervention only if the predictive advantage replicates at "
            "the animal level, the operator stability gate passes on held-out animals, "
            "the perturbation direction is stable, the target/amplitude is in support, "
            "and a pre-registered sham-controlled outcome exists."
        )
        note(
            "These are research proposals, not results. The current Studio does not "
            "predict the outcome of any of them."
        )
        st.markdown(
            " ".join(
                [
                    evidence_badge("SUPPORTED_EXPLORATORY"),
                    badge("ANIMAL-LEVEL REPLICATION PENDING", "danger"),
                    badge("CAUSAL VALIDATION PENDING", "danger"),
                ]
            ),
            unsafe_allow_html=True,
        )
    caption(
        "Sources: reports/master_review/history_timeline.csv, corrections.json, "
        "claims_ledger.json, docs/MASTER_REVIEW_EDUCATIONAL_COURSE.md, "
        "docs/KATO_INTERACTIONS_NEXT_EXPERIMENTS.md."
    )
