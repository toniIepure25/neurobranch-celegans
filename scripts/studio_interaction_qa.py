"""Exercise shared selectors, 3D rendering, timeline controls, and theme switch."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "scientific_studio" / "presentation_polish" / "after" / "interactions"
CHROME = Path(
    r"C:\Users\ComputaCenter\AppData\Local\ms-playwright\chromium-1234\chrome-win64\chrome.exe"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8517")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"generated_utc": datetime.now(UTC).isoformat(), "checks": {}}
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=str(CHROME),
            headless=True,
            args=["--use-gl=swiftshader", "--enable-webgl"],
        )
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.on(
            "console",
            lambda msg: errors.append(f"{msg.type}: {msg.text}") if msg.type == "error" else None,
        )
        page.goto(args.url, wait_until="networkidle", timeout=120_000)
        page.get_by_text("3D Neural Dynamics", exact=True).first.click()
        page.wait_for_timeout(2_000)
        with contextlib.suppress(Exception):
            page.wait_for_selector("canvas", timeout=20_000)
        page.wait_for_timeout(2_000)

        report["checks"]["canvas_count"] = page.locator("canvas").count()
        report["checks"]["plotly_count"] = page.locator(".js-plotly-plot").count()

        page.get_by_role("combobox").nth(0).click()
        page.wait_for_timeout(300)
        page.get_by_role("option", name=re.compile("Recording 2")).click()
        page.wait_for_timeout(1_800)
        recording = page.get_by_role("combobox").nth(0).get_attribute("value") or ""
        report["checks"]["selected_recording"] = recording

        page.get_by_role("combobox").nth(1).click()
        page.wait_for_timeout(300)
        page.get_by_role("option").nth(5).click()
        page.wait_for_timeout(1_800)
        landmark = page.get_by_role("combobox").nth(1).get_attribute("value") or ""
        report["checks"]["selected_landmark"] = landmark

        page.get_by_text("View controls", exact=False).first.click()
        page.wait_for_timeout(600)
        slider = page.get_by_role("slider").nth(0)
        before = slider.get_attribute("aria-valuetext")
        slider.focus()
        slider.press("ArrowRight")
        page.wait_for_timeout(1_200)
        after = page.get_by_role("slider").nth(0).get_attribute("aria-valuetext")
        report["checks"]["history_length_before"] = before
        report["checks"]["history_length_after"] = after
        page.screenshot(path=str(OUT / "dynamics_recording2.png"))

        page.get_by_text("Neural Activity", exact=True).first.click()
        page.wait_for_timeout(2_000)
        activity_recording = page.get_by_role("combobox").nth(0).get_attribute("value") or ""
        activity_landmark = page.get_by_role("combobox").nth(1).get_attribute("value") or ""
        report["checks"]["activity_recording"] = activity_recording
        report["checks"]["activity_landmark"] = activity_landmark
        page.screenshot(path=str(OUT / "activity_shared_selection.png"))

        dark_bg = page.evaluate(
            "() => getComputedStyle(document.documentElement).getPropertyValue('--nb-bg').trim()"
        )
        page.locator("[data-testid='stSidebar']").get_by_text("Light mode", exact=True).first.click()
        page.wait_for_timeout(1_500)
        light_bg = page.evaluate(
            "() => getComputedStyle(document.documentElement).getPropertyValue('--nb-bg').trim()"
        )
        report["checks"]["dark_background"] = dark_bg
        report["checks"]["light_background"] = light_bg
        page.screenshot(path=str(OUT / "activity_light_1920.png"))

        page.get_by_role("button", name=re.compile("Presentation mode")).first.click()
        page.wait_for_selector("text=SLIDE 01 / 10", timeout=30_000)
        page.get_by_role("button", name=re.compile("Exit presentation")).first.click()
        page.wait_for_selector("text=Presentation mode", timeout=30_000)
        report["checks"]["presentation_enter_exit"] = True

        report["checks"]["recording_shared"] = "Recording 2" in activity_recording
        report["checks"]["landmark_shared"] = landmark == activity_landmark
        report["checks"]["slider_changed"] = before != after
        report["checks"]["theme_changed"] = dark_bg != light_bg
        report["checks"]["canvas_rendered"] = report["checks"]["canvas_count"] >= 1
        report["console_errors"] = errors
        browser.close()

    report["status"] = (
        "PASS"
        if all(
            report["checks"][key]
            for key in (
                "recording_shared",
                "landmark_shared",
                "slider_changed",
                "theme_changed",
                "canvas_rendered",
                "presentation_enter_exit",
            )
        )
        and not errors
        else "FAIL"
    )
    (OUT / "interaction_qa.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"], "errors": errors}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
