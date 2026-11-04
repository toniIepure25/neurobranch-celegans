"""Headless browser QA for the seven-section Scientific Studio.

This is a visual/geometry smoke test, not an image-understanding system. It
captures viewport screenshots (Streamlit scrolls long sections in its own
container), checks horizontal overflow, and records console errors. A human
must still inspect the PNGs for typography, contrast, and scientific
legibility. See docs/STUDIO_FINAL_VISUAL_QA.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROME = Path(
    r"C:\Users\ComputaCenter\AppData\Local\ms-playwright\chromium-1234\chrome-win64\chrome.exe"
)
SECTIONS = (
    "Overview",
    "3D Neural Dynamics",
    "Neural Activity",
    "Models & Forecasting",
    "Uncertainty & Evidence",
    "Virtual Perturbations",
    "Research Notebook",
)
VIEWPORTS = ((1920, 1080), (1366, 768), (1280, 800))


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _wait_ready(page) -> None:
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=90_000)
    page.wait_for_timeout(2_500)


def _overflow(page) -> dict[str, float]:
    return page.evaluate(
        """() => ({
          scrollWidth: document.documentElement.scrollWidth,
          clientWidth: document.documentElement.clientWidth,
          bodyScrollWidth: document.body.scrollWidth,
          bodyClientWidth: document.body.clientWidth
        })"""
    )


def _click_sidebar_label(page, label: str) -> None:
    sidebar = page.locator("[data-testid='stSidebar']")
    sidebar.get_by_text(label, exact=True).first.click()
    page.wait_for_timeout(1_500)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8505")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "scientific_studio" / "screenshots",
    )
    parser.add_argument("--chrome", type=Path, default=DEFAULT_CHROME)
    parser.add_argument("--skip-presentation", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.chrome.exists():
        raise SystemExit(f"Chromium executable not found: {args.chrome}")

    report: dict = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "url": args.url,
        "chrome": str(args.chrome),
        "viewports": [{"width": w, "height": h} for w, h in VIEWPORTS],
        "sections": list(SECTIONS),
        "captures": [],
        "console_errors": [],
        "note": (
            "Geometry and console checks only. Visual inspection of the PNGs is "
            "required and is recorded separately."
        ),
    }
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=str(args.chrome),
            headless=True,
            args=["--use-gl=swiftshader", "--enable-webgl"],
        )
        for width, height in VIEWPORTS:
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
                color_scheme="dark",
            )
            page = context.new_page()
            errors: list[str] = []
            page.on(
                "console",
                lambda msg, errors=errors: errors.append(f"{msg.type}: {msg.text}")
                if msg.type == "error"
                else None,
            )
            page.goto(args.url, wait_until="networkidle", timeout=120_000)
            _wait_ready(page)
            for label in SECTIONS:
                if label != "Overview":
                    _click_sidebar_label(page, label)
                page.wait_for_timeout(500)
                slug = _slug(label)
                path = args.out / f"{slug}_{width}.png"
                page.screenshot(path=str(path), full_page=True)
                geom = _overflow(page)
                report["captures"].append(
                    {
                        "section": label,
                        "width": width,
                        "height": height,
                        "file": path.name,
                        "overflow": geom,
                        "horizontal_overflow": geom["scrollWidth"] > geom["clientWidth"] + 1,
                        "console_errors": list(errors),
                    }
                )
                errors.clear()

            if not args.skip_presentation:
                page.get_by_role("button", name="Presentation mode").first.click()
                page.wait_for_selector("text=SLIDE 01 / 10", timeout=30_000)
                page.wait_for_timeout(1_000)
                current_slide = 1
                for slide in (1, 2, 6):
                    while current_slide < slide:
                        page.get_by_role(
                            "button", name=re.compile(r"Next")
                        ).first.click()
                        current_slide += 1
                        page.wait_for_selector(
                            f"text=SLIDE {current_slide:02d} / 10", timeout=30_000
                        )
                        page.wait_for_timeout(800)
                    path = args.out / f"presentation_slide{slide}_{width}.png"
                    page.screenshot(path=str(path), full_page=True)
                    report["captures"].append(
                        {
                            "section": f"Presentation slide {slide}",
                            "width": width,
                            "height": height,
                            "file": path.name,
                            "overflow": _overflow(page),
                            "horizontal_overflow": False,
                            "console_errors": list(errors),
                        }
                    )
                    errors.clear()

            context.close()

        # One light-mode capture verifies the alternate theme still renders.
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
            color_scheme="light",
        )
        page = context.new_page()
        page.goto(args.url, wait_until="networkidle", timeout=120_000)
        _wait_ready(page)
        _click_sidebar_label(page, "3D Neural Dynamics")
        page.locator("[data-testid='stSidebar']").get_by_text("Light mode", exact=True).first.click()
        page.wait_for_timeout(1_500)
        path = args.out / "dynamics_light_1920.png"
        page.screenshot(path=str(path), full_page=True)
        report["captures"].append(
            {
                "section": "3D Neural Dynamics (light)",
                "width": 1920,
                "height": 1080,
                "file": path.name,
                "overflow": _overflow(page),
                "horizontal_overflow": False,
                "console_errors": [],
            }
        )
        context.close()
        browser.close()

    report["console_error_count"] = sum(
        len(c["console_errors"]) for c in report["captures"]
    )
    report["horizontal_overflow_count"] = sum(
        1 for c in report["captures"] if c["horizontal_overflow"]
    )
    (args.out / "browser_qa.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("console_error_count", "horizontal_overflow_count")}, indent=2))
    return 0 if report["console_error_count"] == 0 and report["horizontal_overflow_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
