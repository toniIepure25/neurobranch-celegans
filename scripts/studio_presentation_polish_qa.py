"""Browser QA for the ten-slide presentation polish.

This is geometry, console, and navigation evidence. It does not understand the
pixels it captures; human aesthetic sign-off remains a separate gate.
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
    page.wait_for_timeout(2_000)


def _scroll_main_top(page) -> None:
    page.evaluate(
        """() => {
          const main = document.querySelector('[data-testid="stMain"]');
          if (main) main.scrollTop = 0;
        }"""
    )
    page.wait_for_timeout(150)


def _geometry(page) -> dict:
    return page.evaluate(
        """() => {
          const box = (el) => {
            if (!el) return null;
            const r = el.getBoundingClientRect();
            return {
              top: Math.round(r.top), bottom: Math.round(r.bottom),
              left: Math.round(r.left), right: Math.round(r.right),
              width: Math.round(r.width), height: Math.round(r.height),
              scrollHeight: el.scrollHeight, clientHeight: el.clientHeight,
              scrollWidth: el.scrollWidth, clientWidth: el.clientWidth,
            };
          };
          const shell = document.querySelector('[class*="st-key-nb_slide"]');
          const content = [
            ...document.querySelectorAll(
              '.nb-slide-head, .nb-slide-message, .nb-slide-body, .nb-panel, '
              + '[data-testid="stMetric"], .js-plotly-plot, .nb-status-strip, .nb-caption'
            )
          ];
          const nav = [...document.querySelectorAll('button')]
            .filter((b) => /Previous|Next/.test(b.innerText))
            .map(box);
          const inside = shell
            ? content.every((el) => {
                const r = el.getBoundingClientRect();
                const s = shell.getBoundingClientRect();
                return r.top >= s.top - 2 && r.bottom <= s.bottom + 2;
              })
            : false;
          const clipped = [
            ...document.querySelectorAll(
              '.nb-slide-head h2, .nb-slide-message, .nb-slide-body, .nb-panel, '
              + '[data-testid="stMetricLabel"], [data-testid="stMetricValue"]'
            )
          ]
            .filter((el) => el.scrollHeight > el.clientHeight + 2 || el.scrollWidth > el.clientWidth + 2)
            .map((el) => ({
              text: el.innerText.slice(0, 60),
              scrollHeight: el.scrollHeight,
              clientHeight: el.clientHeight,
              scrollWidth: el.scrollWidth,
              clientWidth: el.clientWidth,
            }));
          const h2 = document.querySelector('.nb-slide-head h2');
          const message = document.querySelector('.nb-slide-message');
          const body = document.querySelector('.nb-slide-body');
          const main = document.querySelector('[data-testid="stMain"]');
          return {
            shell: box(shell),
            shell_class: shell ? shell.className : null,
            content_inside_shell: inside,
            nav: nav,
            nav_in_viewport: nav.every((n) => n && n.bottom <= window.innerHeight + 2),
            clipped,
            fonts: {
              title: h2 ? getComputedStyle(h2).fontSize : null,
              message: message ? getComputedStyle(message).fontSize : null,
              body: body ? getComputedStyle(body).fontSize : null,
            },
            doc_scroll_width: document.documentElement.scrollWidth,
            doc_client_width: document.documentElement.clientWidth,
            main_scroll_width: main ? main.scrollWidth : null,
            main_client_width: main ? main.clientWidth : null,
            viewport: {width: window.innerWidth, height: window.innerHeight},
          };
        }"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8510")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "scientific_studio" / "presentation_polish" / "after",
    )
    parser.add_argument("--chrome", type=Path, default=DEFAULT_CHROME)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sections").mkdir(exist_ok=True)
    if not args.chrome.exists():
        raise SystemExit(f"Chromium executable not found: {args.chrome}")

    report: dict = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "url": args.url,
        "viewports": [{"width": w, "height": h} for w, h in VIEWPORTS],
        "slides": [],
        "sections": [],
        "console_errors": [],
        "note": "Geometry, navigation, and console evidence only; human sign-off remains required.",
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
            _scroll_main_top(page)
            page.screenshot(path=str(args.out / f"overview_{width}.png"))

            for label in SECTIONS:
                if label != "Overview":
                    page.locator("[data-testid='stSidebar']").get_by_text(label, exact=True).first.click()
                    page.wait_for_timeout(900)
                    _scroll_main_top(page)
                path = args.out / "sections" / f"{_slug(label)}_{width}.png"
                page.screenshot(path=str(path), full_page=True)
                geom = _geometry(page)
                report["sections"].append(
                    {
                        "section": label,
                        "width": width,
                        "height": height,
                        "file": str(path.relative_to(args.out)),
                        "horizontal_overflow": geom["doc_scroll_width"] > geom["doc_client_width"] + 1,
                        "console_errors": list(errors),
                    }
                )
                errors.clear()

            page.locator("[data-testid='stSidebar']").get_by_text("Overview", exact=True).first.click()
            page.wait_for_timeout(600)
            page.get_by_role("button", name=re.compile("Presentation mode")).first.click()
            page.wait_for_selector("text=SLIDE 01 / 10", timeout=30_000)
            page.wait_for_timeout(1_000)

            for index in range(10):
                if index:
                    page.get_by_role("button", name=re.compile("Next")).first.click()
                page.wait_for_selector(f"text=SLIDE {index + 1:02d} / 10", timeout=30_000)
                page.wait_for_timeout(900)
                _scroll_main_top(page)
                path = args.out / f"slide_{index + 1:02d}_{width}.png"
                page.screenshot(path=str(path))
                geom = _geometry(page)
                report["slides"].append(
                    {
                        "slide": index + 1,
                        "width": width,
                        "height": height,
                        "file": path.name,
                        **geom,
                        "console_errors": list(errors),
                    }
                )
                errors.clear()

            # Navigation contract: Previous returns to slide 9, Next returns to 10.
            page.get_by_role("button", name=re.compile("Previous")).first.click()
            page.wait_for_selector("text=SLIDE 09 / 10", timeout=30_000)
            page.get_by_role("button", name=re.compile("Next")).first.click()
            page.wait_for_selector("text=SLIDE 10 / 10", timeout=30_000)

            page.get_by_role("button", name=re.compile("Exit presentation")).first.click()
            page.wait_for_selector("text=Presentation mode", timeout=30_000)
            page.wait_for_timeout(800)
            report["sections"].append(
                {
                    "section": "Presentation exit",
                    "width": width,
                    "height": height,
                    "file": None,
                    "horizontal_overflow": False,
                    "console_errors": list(errors),
                }
            )
            errors.clear()
            context.close()

        # Light mode: all ten slides at 1920 and one Overview capture.
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
            color_scheme="light",
        )
        page = context.new_page()
        errors = []
        page.on(
            "console",
            lambda msg, errors=errors: errors.append(f"{msg.type}: {msg.text}")
            if msg.type == "error"
            else None,
        )
        page.goto(args.url, wait_until="networkidle", timeout=120_000)
        _wait_ready(page)
        page.locator("[data-testid='stSidebar']").get_by_text("Light mode", exact=True).first.click()
        page.wait_for_timeout(1_200)
        page.screenshot(path=str(args.out / "overview_light_1920.png"))
        page.get_by_role("button", name=re.compile("Presentation mode")).first.click()
        page.wait_for_selector("text=SLIDE 01 / 10", timeout=30_000)
        page.wait_for_timeout(900)
        for index in range(10):
            if index:
                page.get_by_role("button", name=re.compile("Next")).first.click()
            page.wait_for_selector(f"text=SLIDE {index + 1:02d} / 10", timeout=30_000)
            page.wait_for_timeout(800)
            _scroll_main_top(page)
            path = args.out / f"slide_{index + 1:02d}_light_1920.png"
            page.screenshot(path=str(path))
            geom = _geometry(page)
            report["slides"].append(
                {
                    "slide": index + 1,
                    "width": 1920,
                    "height": 1080,
                    "theme": "light",
                    "file": path.name,
                    **geom,
                    "console_errors": list(errors),
                }
            )
            errors.clear()
        context.close()
        browser.close()

    report["console_error_count"] = sum(len(c["console_errors"]) for c in report["sections"] + report["slides"])
    report["horizontal_overflow_count"] = sum(
        1 for c in report["sections"] if c.get("horizontal_overflow")
    )
    report["content_outside_shell_count"] = sum(
        1 for s in report["slides"] if not s["content_inside_shell"]
    )
    report["nav_offscreen_count"] = sum(1 for s in report["slides"] if not s["nav_in_viewport"])
    report["clipped_count"] = sum(len(s["clipped"]) for s in report["slides"])
    (args.out / "browser_polish_qa.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "console_error_count",
                    "horizontal_overflow_count",
                    "content_outside_shell_count",
                    "nav_offscreen_count",
                    "clipped_count",
                )
            },
            indent=2,
        )
    )
    clean = (
        report["console_error_count"] == 0
        and report["horizontal_overflow_count"] == 0
        and report["content_outside_shell_count"] == 0
        and report["nav_offscreen_count"] == 0
        and report["clipped_count"] == 0
    )
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
