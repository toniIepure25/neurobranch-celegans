"""Compose human-review contact sheets from browser QA screenshots.

The sheets are a navigation aid for a human reviewer. They are not an
automated visual-quality judgment and must not be cited as one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "reports" / "scientific_studio" / "screenshots"
OUT = ROOT / "reports" / "scientific_studio" / "final_audit"


def _font(size: int):
    for name in ("segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _grid(files: list[tuple[str, Path]], *, columns: int, crop_h: int, out: Path) -> None:
    thumb_w = 560
    label_h = 42
    rows = (len(files) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_w, rows * (crop_h + label_h)), "#101820")
    draw = ImageDraw.Draw(canvas)
    font = _font(20)
    for index, (label, path) in enumerate(files):
        image = Image.open(path).convert("RGB")
        scale = thumb_w / image.width
        resized = image.resize((thumb_w, max(1, int(image.height * scale))))
        crop = resized.crop((0, 0, thumb_w, min(crop_h, resized.height)))
        x = (index % columns) * thumb_w
        y = (index // columns) * (crop_h + label_h)
        canvas.paste(crop, (x, y + label_h))
        draw.text((x + 8, y + 8), label, fill="#e8f1f5", font=font)
    canvas.save(out)
    print(f"wrote {out.relative_to(ROOT)} ({canvas.width}x{canvas.height})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads((SHOTS / "browser_qa.json").read_text(encoding="utf-8"))
    sections = [
        (f"{c['section']} · {c['width']}", SHOTS / c["file"])
        for c in report["captures"]
        if c["width"] == 1920 and not c["section"].startswith("Presentation")
    ]
    _grid(sections, columns=2, crop_h=720, out=args.out / "contact_sheet_sections_1920.png")
    presentations = [
        (f"{c['section']} · {c['width']}", SHOTS / c["file"])
        for c in report["captures"]
        if c["section"].startswith("Presentation")
    ]
    _grid(presentations, columns=2, crop_h=620, out=args.out / "contact_sheet_presentation.png")


if __name__ == "__main__":
    main()
