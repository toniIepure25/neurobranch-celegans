"""Render the handoff memo to a two-page PDF and the talk to HTML/PDF slides.

The editable sources are the Markdown files in ``docs/``. This script only
renders them; it does not change any scientific number.

Usage:
    python scripts/build_handoff_pdf.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "simulator" / "supervisor_handoff"
MEMO_MD = ROOT / "docs" / "HANDOFF_RESEARCH_MEMO.md"
TALK_MD = ROOT / "docs" / "HANDOFF_PRESENTATION.md"
MEMO_PDF = OUT / "NeuroBranch_Handoff_Memo.pdf"
TALK_HTML = OUT / "NeuroBranch_Handoff_Presentation.html"
TALK_PDF = OUT / "NeuroBranch_Handoff_Presentation.pdf"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION \u2014 NOT CAUSALLY VALIDATED."


def chrome() -> Path:
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return candidate
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return Path(found)
    raise FileNotFoundError("No Chrome or Edge executable found for PDF export")


def md_to_html(text: str) -> str:
    return markdown.markdown(
        text,
        extensions=["tables", "sane_lists", "nl2br"],
        output_format="html5",
    )


def page_count(pdf: Path) -> int:
    data = pdf.read_bytes()
    return len(re.findall(rb"/Type\s*/Page\b", data))


def print_pdf(html_path: Path, pdf_path: Path, extra: list[str] | None = None) -> None:
    args = [
        str(chrome()),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=5000",
    ]
    if extra:
        args.extend(extra)
    args.append(f"--print-to-pdf={pdf_path}")
    args.append(html_path.resolve().as_uri())
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


MEMO_CSS = """
@page { size: A4; margin: 13mm 14mm 12mm 14mm; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: white; color: #17202a; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; }
.memo { max-width: 100%; }
h1 {
  font-size: 15.2pt; line-height: 1.12; margin: 0 0 5pt 0;
  color: #0b3d4a; letter-spacing: -0.15pt;
}
h1 + p { margin-top: 0; }
p { font-size: 8.55pt; line-height: 1.22; margin: 0 0 3.4pt 0; text-align: justify; }
h2 {
  font-size: 9.7pt; line-height: 1.2; margin: 8pt 0 2.2pt 0;
  color: #0b3d4a; border-bottom: 0.7pt solid #d5dbdb; padding-bottom: 1pt;
}
strong { color: #17202a; }
em { color: #17202a; }
ul, ol { margin: 2pt 0 3.5pt 15pt; padding: 0; }
li { font-size: 8.45pt; line-height: 1.19; margin: 0 0 1.6pt 0; text-align: justify; }
table {
  width: 100%; border-collapse: collapse; margin: 3pt 0 5pt 0;
  font-size: 7.15pt; line-height: 1.12;
}
th {
  background: #eaf2f1; color: #0b3d4a; text-align: left;
  border-bottom: 0.8pt solid #a2d9ce; padding: 2pt 3pt;
}
td { border-bottom: 0.4pt solid #e5e8e8; padding: 1.8pt 3pt; vertical-align: top; }
tr:nth-child(even) td { background: #fbfcfc; }
blockquote {
  margin: 4pt 0; padding: 3pt 7pt; border-left: 2.2pt solid #b9770e;
  background: #fef9e7; font-size: 7.9pt;
}
blockquote p { margin: 0; font-size: 7.9pt; text-align: left; }
a { color: #1f618d; text-decoration: none; }
hr { border: 0; border-top: 0.6pt solid #d5dbdb; margin: 6pt 0; }
code { font-family: Consolas, monospace; font-size: 7.5pt; }
.references { font-size: 7.15pt; }
.references p { font-size: 7.15pt; line-height: 1.16; margin-bottom: 2.2pt; }
.footer {
  margin-top: 4pt; border-top: 0.6pt solid #d5dbdb; padding-top: 2.5pt;
  font-size: 6.5pt; color: #5d6d7e;
}
"""


def build_memo() -> int:
    text = MEMO_MD.read_text(encoding="utf-8")
    body = md_to_html(text)
    body = body.replace("<h2>References</h2>", '<h2 class="references">References</h2>')
    body = body.replace(
        "</body>",
        "",
    )
    # Use a decreasing font scale until the memo fits exactly two A4 pages.
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "memo.html"
        for scale in [1.00, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76]:
            html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>NeuroBranch research memo</title>
<style>{MEMO_CSS}
body {{ font-size: {8.55 * scale:.3f}pt; }}
h1 {{ font-size: {15.2 * scale:.3f}pt; }}
h2 {{ font-size: {9.7 * scale:.3f}pt; }}
p {{ font-size: {8.55 * scale:.3f}pt; }}
li {{ font-size: {8.45 * scale:.3f}pt; }}
table {{ font-size: {7.15 * scale:.3f}pt; }}
.references p {{ font-size: {7.15 * scale:.3f}pt; }}
</style></head><body><div class="memo">{body}
<div class="footer">NeuroBranch internal handoff · exploratory repeated-session reanalysis · not a systematic review · {WARNING}</div>
</div></body></html>"""
            html_path.write_text(html, encoding="utf-8")
            print_pdf(html_path, MEMO_PDF)
            pages = page_count(MEMO_PDF)
            print(f"memo scale {scale:.2f}: {pages} pages")
            if pages == 2:
                return pages
            if pages < 2:
                # Enlarge slightly on the next pass; start at 1.00 so this is rare.
                continue
        pages = page_count(MEMO_PDF)
        if pages > 2:
            raise RuntimeError(f"memo did not fit two pages (last render {pages})")
        return pages


TALK_CSS = """
@page { size: 13.333in 7.5in; margin: 0; }
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; background: #0d1b1e; color: #17202a;
  font-family: "Segoe UI", Calibri, Arial, sans-serif;
}
.deck { width: 100vw; height: 100vh; overflow: hidden; position: relative; }
.slide {
  display: none; width: 100vw; height: 100vh; padding: 5.2vh 5.2vw 4.2vh 5.2vw;
  background: #fbfcfc; position: relative; flex-direction: column;
  justify-content: flex-start;
}
.slide.active { display: flex; }
.slide.cover {
  background:
    radial-gradient(circle at 82% 18%, rgba(162,217,206,0.50), transparent 34%),
    linear-gradient(135deg, #0b3d4a 0%, #102a33 62%, #17202a 100%);
  color: white; justify-content: center;
}
.cover .kicker { color: #a2d9ce; letter-spacing: 0.14em; font-size: 1.05vw; font-weight: 700; }
.cover h1 { color: white; font-size: 3.25vw; line-height: 1.08; max-width: 84%; margin: 2.0vh 0 2.2vh 0; }
.cover .sub { color: #d5dbdb; font-size: 1.3vw; line-height: 1.35; max-width: 66%; }
.cover .labels { margin-top: 3.2vh; display: flex; gap: 0.8vw; flex-wrap: wrap; }
.cover .label {
  border: 1px solid rgba(255,255,255,0.36); border-radius: 999px;
  padding: 0.7vh 1.0vw; color: white; font-size: 0.9vw;
}
.slide h2 {
  font-size: 2.05vw; line-height: 1.13; color: #0b3d4a; margin: 0 0 1.4vh 0;
  letter-spacing: -0.02em;
}
.slide h2 .num { color: #117a65; margin-right: 0.7vw; }
.slide-body { font-size: 1.23vw; line-height: 1.35; flex: 1; min-height: 0; }
.slide-body p { margin: 0 0 1.0vh 0; }
.slide-body ul, .slide-body ol { margin: 0 0 1.0vh 1.5vw; padding: 0; }
.slide-body li { margin: 0 0 0.65vh 0; }
.slide-body strong { color: #0b3d4a; }
.slide-body table { border-collapse: collapse; width: 100%; font-size: 0.92vw; margin: 1.0vh 0; }
.slide-body th { background: #eaf2f1; color: #0b3d4a; text-align: left; padding: 0.65vh 0.65vw; border-bottom: 1px solid #a2d9ce; }
.slide-body td { padding: 0.55vh 0.65vw; border-bottom: 1px solid #e5e8e8; vertical-align: top; }
.slide-body tr:nth-child(even) td { background: #f6f9f8; }
.slide-body blockquote {
  margin: 1.2vh 0; padding: 1.1vh 1.2vw; border-left: 4px solid #b9770e;
  background: #fef9e7; font-size: 1.18vw;
}
.slide-body blockquote p { margin: 0; }
.figure { margin-top: 1.1vh; text-align: center; }
.figure img { max-height: 30vh; max-width: 92%; object-fit: contain; border: 1px solid #e5e8e8; background: white; }
.figure .caption { font-size: 0.78vw; color: #5d6d7e; margin-top: 0.4vh; }
.notes {
  display: none; margin-top: 1.2vh; padding: 1.2vh 1.2vw; background: #eef6f5;
  border-left: 4px solid #117a65; font-size: 0.98vw; line-height: 1.35; color: #17202a;
  max-height: 27vh; overflow: auto;
}
.deck.show-notes .notes { display: block; }
.notes p { margin: 0 0 0.5vh 0; }
.chrome {
  position: fixed; left: 0; right: 0; bottom: 0; height: 4.2vh;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 1.4vw; background: rgba(13,27,30,0.90); color: #d5dbdb;
  font-size: 0.78vw; z-index: 20;
}
.chrome button {
  background: transparent; color: white; border: 1px solid rgba(255,255,255,0.30);
  border-radius: 4px; padding: 0.35vh 0.7vw; cursor: pointer; font-size: 0.78vw;
}
.chrome button:hover { background: rgba(255,255,255,0.12); }
.progress { position: fixed; top: 0; left: 0; height: 0.42vh; background: #117a65; z-index: 30; width: 0; }
.warning {
  position: fixed; left: 1.4vw; bottom: 4.9vh; z-index: 25;
  background: #922b21; color: white; padding: 0.55vh 0.8vw;
  font-size: 0.72vw; letter-spacing: 0.02em; border-radius: 3px;
}
@media print {
  html, body { background: white; }
  .deck { width: auto; height: auto; overflow: visible; }
  .slide {
    display: flex !important; width: 13.333in; height: 7.5in;
    page-break-after: always; break-after: page;
  }
  .chrome, .progress, .warning { display: none !important; }
  .notes { display: none !important; }
}
"""


def split_slides(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    parts = re.split(r"\n## Slide\s+", text)
    slides = []
    for part in parts[1:]:
        slides.append("## Slide " + part.strip())
    return title, slides


def build_talk() -> None:
    text = TALK_MD.read_text(encoding="utf-8")
    title, slides = split_slides(text)
    title_html = re.sub(r"^<p>|</p>$", "", md_to_html(title))
    figures = {
        5: ("fig02_central_result.png", "Paired H20 contrasts in the fixed 10-D PCA."),
        7: ("fig03_calibration.png", "Full-neuron and 10-D PCA coverage; the 50% level fails."),
        9: ("fig05_proposed_scope.png", "Two bounded 6-ECTS work plans."),
    }
    sections = []
    cover = f"""
<section class="slide cover active" data-slide="0">
  <div class="kicker">NEUROBRANCH · MSc SUPERVISOR HANDOFF</div>
  <h1>{title_html}</h1>
  <div class="sub">Exploratory repeated-session reanalysis of <em>C. elegans</em> population
  calcium dynamics, joint forecasting and model-dependent virtual perturbations.</div>
  <div class="labels">
    <span class="label">5 Kato sessions, not 5 animals</span>
    <span class="label">Fixed 10-D PCA</span>
    <span class="label">Calibration not established</span>
    <span class="label">Biological validation NOT_EVALUABLE</span>
  </div>
</section>"""
    sections.append(cover)
    for index, raw in enumerate(slides, start=1):
        heading_match = re.match(r"## Slide\s+(\d+)\s*\u2014\s*(.*)", raw.splitlines()[0])
        if heading_match:
            num = heading_match.group(1)
            heading = heading_match.group(2)
            body_md = "\n".join(raw.splitlines()[1:]).strip()
        else:
            num = str(index)
            heading = f"Slide {index}"
            body_md = raw
        notes_html = ""
        marker = "**Speaker notes**"
        if marker in body_md:
            body_part, notes_part = body_md.split(marker, 1)
            body_md = body_part.strip()
            notes_html = f'<div class="notes"><strong>Speaker notes</strong>{md_to_html(notes_part)}</div>'
        body_html = md_to_html(body_md)
        body_html = re.sub(r"<h2>.*?</h2>", "", body_html, flags=re.S)
        body_html = body_html.replace("<p><strong>On slide</strong></p>", "")
        fig_html = ""
        if index in figures:
            name, caption = figures[index]
            fig_html = (
                f'<div class="figure"><img src="figures/{name}" alt="{caption}">'
                f'<div class="caption">{caption}</div></div>'
            )
        sections.append(
            f"""
<section class="slide" data-slide="{index}">
  <h2><span class="num">{num.zfill(2)}</span>{heading}</h2>
  <div class="slide-body">{body_html}{fig_html}{notes_html}</div>
</section>"""
        )
    total = len(sections)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{TALK_CSS}</style>
</head><body>
<div class="progress" id="progress"></div>
<div class="deck" id="deck">
{''.join(sections)}
</div>
<div class="warning">MODEL-DEPENDENT VIRTUAL PERTURBATION \u2014 NOT CAUSALLY VALIDATED.</div>
<div class="chrome">
  <div><button id="prev">← Prev</button> <button id="next">Next →</button>
    <button id="toggle-notes">Notes (N)</button></div>
  <div id="counter">1 / {total}</div>
  <div>Arrow keys · Home/End · N notes</div>
</div>
<script>
const slides = Array.from(document.querySelectorAll('.slide'));
const progress = document.getElementById('progress');
const counter = document.getElementById('counter');
const deck = document.getElementById('deck');
let current = 0;
function show(i) {{
  current = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((s, j) => s.classList.toggle('active', j === current));
  progress.style.width = ((current + 1) / slides.length * 100) + '%';
  counter.textContent = (current + 1) + ' / ' + slides.length;
  history.replaceState(null, '', '?slide=' + (current + 1));
}}
document.getElementById('prev').onclick = () => show(current - 1);
document.getElementById('next').onclick = () => show(current + 1);
document.getElementById('toggle-notes').onclick = () => {{
  deck.classList.toggle('show-notes');
  document.getElementById('toggle-notes').textContent =
    deck.classList.contains('show-notes') ? 'Hide notes (N)' : 'Notes (N)';
}};
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') {{ e.preventDefault(); show(current + 1); }}
  if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{ e.preventDefault(); show(current - 1); }}
  if (e.key === 'Home') {{ e.preventDefault(); show(0); }}
  if (e.key === 'End') {{ e.preventDefault(); show(slides.length - 1); }}
  if (e.key.toLowerCase() === 'n') {{ document.getElementById('toggle-notes').click(); }}
}});
const params = new URLSearchParams(location.search);
const start = parseInt(params.get('slide') || '1', 10) - 1;
show(Number.isFinite(start) ? start : 0);
</script>
</body></html>"""
    TALK_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {TALK_HTML.relative_to(ROOT)} ({total} slides)")

    try:
        print_pdf(TALK_HTML, TALK_PDF)
        print(f"wrote {TALK_PDF.relative_to(ROOT)} ({page_count(TALK_PDF)} pages)")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"warning: presentation PDF export failed: {exc}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pages = build_memo()
    print(f"wrote {MEMO_PDF.relative_to(ROOT)} ({pages} pages)")
    build_talk()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_supervisor_handoff.py"), "--manifest-only"],
        check=True,
    )


if __name__ == "__main__":
    main()
