import argparse
import subprocess
import sys
import time
from pathlib import Path

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

        # Determine latest exit_counts mtime before click
        signals_dir = Path("data_cache/signals")

        def latest_mtime() -> float:
            try:
                if not signals_dir.exists():
                    return 0.0
                files = list(signals_dir.glob("exit_counts_*.json"))
                if not files:
                    return 0.0
                return max(p_.stat().st_mtime for p_ in files)
            except Exception:
                return 0.0

        before_mtime = latest_mtime()

        # Try clicking a run button containing '本日のシグナル' or '実行'
        clicked = False
        selectors = [
            ("role", "button", "本日のシグナル"),
            ("text", None, "本日のシグナル"),
            ("text", None, "実行"),
            ("text", None, "Run"),
            ("aria", None, "reload"),
            ("css", None, "button"),
        ]
        for kind, typ, txt in selectors:
            try:
                if kind == "role":
                    btn = page.get_by_role(typ, name=txt)
                    btn.click(timeout=5000)
                    clicked = True
                    print(f"Clicked button by role name: {txt}")
                    break
                else:
                    if kind == "text":
                        el = page.locator(f"text={txt}").first
                        el.click(timeout=5000)
                    elif kind == "aria":
                        # try attribute-based selectors
                        el = page.locator(f"[aria-label*='{txt}']").first
                        el.click(timeout=5000)
                    elif kind == "css":
                        # try clicking visible button elements with matching inner
                        # text heuristics
                        els = page.locator("button").all()
                        clicked_local = False
                        for e in els:
                            try:
                                txt_inner = e.inner_text() or ""
                            except Exception:
                                txt_inner = ""
                            if any(
                                k in txt_inner
                                for k in ("実行", "本日のシグナル", "再読み込み", "Run")
                            ):
                                e.click(timeout=2000)
                                clicked_local = True
                                break
                        if not clicked_local:
                            # try first button as last resort
                            els[0].click(timeout=2000)

                    else:
                        el = page.locator(f"text={txt}").first
                        el.click(timeout=5000)
                    clicked = True
                    print(f"Clicked element with text: {txt}")
                    break
            except Exception:
                continue

        if not clicked:
            print("No run-like button found to click")

        # Wait for new exit_counts file to appear (polling)
        found_new = False
        new_mtime = None
        start = time.time()
        timeout = wait_timeout
        while time.time() - start < timeout:
            time.sleep(0.5)
            after = latest_mtime()
            if after > before_mtime:
                found_new = True
                new_mtime = after
                break

        # capture screenshot and extract page text for diff lines
        page.screenshot(path=str(out_path), full_page=True)
        print(f"Saved screenshot: {out_path}")
        try:
            body = page.inner_text("body")
        except Exception:
            body = page.content()

        # look for lines like 'systemX: a → b (+n)'
        diffs = []
        for line in body.splitlines():
            if "→" in line and (
                "system" in line.lower() or "システム" in line or "system" in line
            ):
                s = line.strip()
                if s:
                    diffs.append(s)

        if found_new:
            try:
                if "new_mtime" in locals():
                    print(f"Detected new exit_counts (mtime={new_mtime})")
                else:
                    print("Detected new exit_counts")
            except Exception:
                print("Detected new exit_counts")
        else:
            print("No new exit_counts detected within timeout")
            # Fallback behaviour: run the runner script directly to ensure
            # exit_counts are generated
            try:
                print("Fallback: launching runner script to produce exit_counts...")
                cmd = [
                    sys.executable,
                    "scripts/run_all_systems_today.py",
                    "--test-mode",
                    "sample",
                    "--skip-external",
                    "--save-csv",
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                print("Runner finished, rc=", proc.returncode)
                if proc.stdout:
                    print(proc.stdout.splitlines()[-10:])
                # recheck files
                after = latest_mtime()
                if after > before_mtime:
                    found_new = True
                    new_mtime = after
            except Exception as e:
                print("Fallback runner error:", e)

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
