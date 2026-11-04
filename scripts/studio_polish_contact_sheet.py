"""Build human-review contact sheets for the presentation polish.

The sheets are navigation aids for a human reviewer, not an automated visual
quality judgment.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports" / "scientific_studio" / "presentation_polish"
AFTER = BASE / "after"
BEFORE = BASE / "before"


def _font(size: int):
    for name in ("segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _grid(items: list[tuple[str, Path]], *, columns: int, out: Path, thumb_w: int = 620) -> None:
    label_h = 40
    rows = (len(items) + columns - 1) // columns
    cell_h = int(thumb_w * 1080 / 1920) + label_h
    canvas = Image.new("RGB", (columns * thumb_w, rows * cell_h), "#101820")
    draw = ImageDraw.Draw(canvas)
    font = _font(20)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        scale = thumb_w / image.width
        resized = image.resize((thumb_w, max(1, int(image.height * scale))))
        crop = resized.crop((0, 0, thumb_w, min(int(thumb_w * 1080 / 1920), resized.height)))
        x = (index % columns) * thumb_w
        y = (index // columns) * cell_h
        canvas.paste(crop, (x, y + label_h))
        draw.text((x + 8, y + 8), label, fill="#e8f1f5", font=font)
    canvas.save(out)
    print(f"wrote {out.relative_to(ROOT)} ({canvas.width}x{canvas.height})")


def main() -> None:
    dark = [(f"Slide {i:02d}", AFTER / f"slide_{i:02d}_1920.png") for i in range(1, 11)]
    light = [(f"Slide {i:02d} · light", AFTER / f"slide_{i:02d}_light_1920.png") for i in range(1, 11)]
    _grid(dark, columns=2, out=AFTER / "contact_sheet_all_slides_1920.png")
    _grid(light, columns=2, out=AFTER / "contact_sheet_all_slides_light_1920.png")
    _grid(
        [
            ("Before · slide 2", BEFORE / "presentation_slide2_1920.png"),
            ("After · slide 2", AFTER / "slide_02_1920.png"),
        ],
        columns=2,
        out=AFTER / "contact_sheet_slide2_before_after.png",
    )
    _grid(
        [
            ("Before · Overview", BEFORE / "overview_1920.png"),
            ("After · Overview", AFTER / "overview_1920.png"),
        ],
        columns=2,
        out=AFTER / "contact_sheet_overview_before_after.png",
    )


if __name__ == "__main__":
    main()
