"""Visit the hosted app so Streamlit Community Cloud does not put it to sleep.

Run by .github/workflows/keep-awake.yml every few hours. Opens the app in a
headless browser; if the "app has gone to sleep" page shows, clicks the button
that wakes it. Then waits until the app itself (its login page) has loaded,
and fails if it has not, so a problem shows up in the Actions tab.

No password is entered: loading the login page counts as a visit.
"""
from __future__ import annotations

import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "https://eu-law-assistant.streamlit.app/")
WAKE_BUTTON = re.compile(r"get this app back up", re.IGNORECASE)
APP_TEXT = "EU LexRef"  # shown on the login page and in the app


def app_loaded(page) -> bool:
    """True once the app's own page is drawn (Cloud shows it inside a frame)."""
    for frame in page.frames:
        try:
            if APP_TEXT in frame.locator("body").inner_text(timeout=2000):
                return True
        except Exception:
            continue
    return False


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(8_000)

        wake = page.get_by_role("button", name=WAKE_BUTTON)
        if wake.count():
            print("The app was asleep: waking it.")
            wake.first.click()
        else:
            print("The app was awake.")

        deadline = time.monotonic() + 240  # waking up takes about a minute
        while time.monotonic() < deadline:
            if app_loaded(page):
                print(f"The app is up: {URL}")
                browser.close()
                return 0
            page.wait_for_timeout(5_000)
        print(f"The app did not load within 4 minutes: {URL}", file=sys.stderr)
        page.screenshot(path="keep-awake-failure.png", full_page=True)
        browser.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
