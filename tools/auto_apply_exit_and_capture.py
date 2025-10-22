from pathlib import Path
import time
import argparse

from playwright.sync_api import sync_playwright


def run(url: str, out: str, wait_timeout: int = 10) -> int:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        # launch headful so Streamlit UI elements render consistently
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-gpu",
            ],
        )
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        start = time.time()
        while True:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=5000)
                break
            except Exception:
                if time.time() - start > wait_timeout:
                    print(f"ERROR: UI not reachable within {wait_timeout}s")
                    browser.close()
                    return 2
                time.sleep(0.5)

        print(f"Opened {url}")

        # Try clicking a button labeled '再読み込み'
        clicked = False
        try:
            # prefer role-based lookup when possible
            try:
                btn = page.get_by_role("button", name="再読み込み")
                btn.click(timeout=5000)
                clicked = True
            except Exception:
                try:
                    el = page.locator("text=再読み込み").first
                    el.click(timeout=5000)
                    clicked = True
                except Exception:
                    clicked = False
        except Exception:
            clicked = False

        if clicked:
            print("Clicked 再読み込み button")
        else:
            print("再読み込み button not found or not clickable")

        # Wait briefly for any '適用しました' message to appear
        found = False
        try:
            page.wait_for_selector("text=適用しました", timeout=8000)
            found = True
        except Exception:
            found = False

        # capture screenshot and extract page text for diff lines
        page.screenshot(path=str(out_path), full_page=True)
        print(f"Saved screenshot: {out_path}")
        body = page.inner_text("body")
        # look for lines like 'systemX: a → b (+n)'
        diffs = []
        for line in body.splitlines():
            if "→" in line and ("system" in line.lower() or "システム" in line):
                s = line.strip()
                if s:
                    diffs.append(s)

        if found:
            print("Found confirmation message '適用しました'")
        else:
            print("No explicit confirmation message found within timeout")

        if diffs:
            print("Detected diff lines:")
            for d in diffs:
                print(d)
        else:
            print("No diff lines detected on page text")

        browser.close()
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--output", default="screenshots/auto_reload.png")
    parser.add_argument("--wait", type=int, default=15)
    args = parser.parse_args()
    rc = run(args.url, args.output, args.wait)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
