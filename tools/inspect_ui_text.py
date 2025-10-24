import sys
import time

from playwright.sync_api import sync_playwright


def main(url: str = "http://localhost:8501") -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        # wait for Streamlit app root or give it some time to render
        try:
            page.wait_for_selector('[data-testid="stApp"]', timeout=10000)
        except Exception:
            # fallback small sleep
            time.sleep(1)
        try:
            body = page.inner_text("body")
        except Exception:
            body = page.content()
        print("--- BODY SNIPPET ---")
        for line in body.splitlines()[:400]:
            print(line)
        print("--- BUTTONS ---")
        try:
            buttons = page.locator("button").all()
            for i, b in enumerate(buttons[:50]):
                try:
                    txt = b.inner_text()
                except Exception:
                    txt = "<no-text>"
                print(f"[{i}] {txt}")
        except Exception as e:
            print("error listing buttons:", e)
        browser.close()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
    main(url)
