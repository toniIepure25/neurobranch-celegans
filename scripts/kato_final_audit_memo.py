"""Build a two-page supervisor memo PDF from the audited numbers."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports" / "simulator" / "kato_final_audit"
OUT = AUDIT / "KATO_FINAL_AUDIT_MEMO.pdf"

SECTIONS = [
    (
        "What was asked",
        "Is the positive Kato WT_NoStim cross-neuron result computationally reproducible, "
        "methodologically comparable to its controls, and safe to present without moving "
        "from prediction to biological mechanism?",
    ),
    (
        "Evidence base",
        "Five repeated WT_NoStim sessions, 15,662 frames, 175 fixed H20 landmarks, native "
        "panels 103/127/123/118/121. Ten-component PCA was fitted on training frames only. "
        "These are repeated recordings, not five confirmed independent animals. Test tails "
        "were reused across project phases.",
    ),
    (
        "Reproduction",
        "A complete clean run of the official entry point finished normally in 2,005.9 s "
        "with empty stderr. All 17 compared artifacts agree within 1e-8 (maximum absolute "
        "difference 4.44e-16); all eight gate statuses match. Status: "
        "FULL_END_TO_END_REPRODUCTION_PASSED. The historical run itself was interrupted "
        "and post-processed; the clean rerun is the evidence.",
    ),
    (
        "Primary positive result",
        "B3 population energy 2.7001; B5-10 own+factors energy 2.9646; B5 - B3 = +0.2644; "
        "block-5 95% interval [+0.0750, +0.4031]; B3 better at 64.0% of 175 landmarks. "
        "Positive means B3 is better because energy is lower-is-better.",
    ),
    (
        "What survives",
        "The sign is positive at blocks 3/5/10/20, in four of five sessions, and at H5, H10 "
        "and H20. Leave-one-session-out pooled means stay positive. H1 reverses (B5 better), "
        "session 2 reverses, and PCA-10 MSE and neuron-space intervals include zero.",
    ),
    (
        "What does not survive scrutiny",
        "G4 is not a fully prespecified confirmatory PASS: the sign rule was clarified in the "
        "same commit as the results, with no independent pre-evaluation protocol. The 51/49 "
        "split is an ordered B4m->B5->B3 decomposition, not a unique variance partition; "
        "with B1m it is 23.5/76.5. The ten-factor control retains all ten evaluation "
        "dimensions but cannot represent B3's arbitrary other-neuron residual rows; B3 also "
        "has 7.96-9.95x more parameters, so information versus capacity is unresolved.",
    ),
    (
        "Biological boundary",
        "No stable pairwise interaction graph: off-diagonal coefficient correlation is 0.052. "
        "No synapse, command neuron, causal connectivity or stimulation target. Virtual "
        "perturbations are mathematical (+1 SD) displacements with cross-fit response cosine "
        "0.556 and sign reversals. Biological intervention validity: NOT_EVALUABLE.",
    ),
    (
        "Question for Denis",
        "For an MSc, is an independent-cohort, animal-level replication of the B5-controlled "
        "H20 energy contrast sufficient, or does the project need matched intervention data "
        "before the predictive result is scientifically useful?",
    ),
]


def main() -> None:
    width = 88
    pages: list[list[tuple[str, list[str]]]] = [[]]
    y = 0.90
    for heading, body in SECTIONS:
        lines = textwrap.wrap(body, width=width)
        needed = 0.035 + 0.024 * len(lines) + 0.02
        if y - needed < 0.09:
            pages.append([])
            y = 0.90
        pages[-1].append((heading, lines))
        y -= needed
    if len(pages) != 2:
        raise RuntimeError(f"memo layout produced {len(pages)} pages, expected 2")

    with PdfPages(OUT) as pdf:
        for number, page in enumerate(pages, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.95, "Kato final audit: supervisor memo", fontsize=16, weight="bold")
            fig.text(
                0.06,
                0.918,
                "Exploratory predictive evidence; not a confirmatory or mechanistic claim",
                fontsize=9.5,
                color="#444444",
            )
            y = 0.875
            for heading, lines in page:
                fig.text(0.06, y, heading, fontsize=10.5, weight="bold", color="#1f4e5f")
                y -= 0.024
                for line in lines:
                    fig.text(0.06, y, line, fontsize=9.0)
                    y -= 0.024
                y -= 0.012
            fig.text(
                0.06,
                0.045,
                "PROBABILISTIC CALIBRATION NOT ESTABLISHED  |  "
                "MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED.",
                fontsize=7.2,
                color="#7a1f2b",
            )
            fig.text(0.94, 0.045, f"page {number}/2", fontsize=7.5, ha="right")
            pdf.savefig(fig)
            plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
