"""Streamlit UI のスクリーンショットを自動取得するツール。

Playwright を使用して、実行中の Streamlit アプリのスクリーンショットを保存します。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


def capture_streamlit_screenshot(
    url: str,
    output_path: Path,
    wait_seconds: int = 3,
    click_button: str | None = None,
    wait_after_click: int = 15,
    scroll_to_bottom: bool = True,
) -> bool:
    """Streamlit UI のスクリーンショットを取得。

    Args:
        url: Streamlit アプリの URL (例: http://localhost:8501)
        output_path: 保存先パス
        wait_seconds: ページ読み込み待機時間（秒）
        click_button: クリックするボタンのテキスト（例: "Run Today Signals"）
        wait_after_click: ボタンクリック後の待機時間（秒）
        scroll_to_bottom: 最下部までスクロールするか（デフォルト: True）

    Returns:
        成功した場合 True
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright が未インストールです。以下を実行してください:")
        print("  pip install playwright")
        print("  playwright install chromium")
        return False

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})

            print(f"Loading: {url}")
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(wait_seconds)

            # ボタンクリック（オプション）
            if click_button:
                print(f"Clicking button: '{click_button}'")
                try:
                    # Streamlit のボタンを探してクリック
                    button = page.get_by_role("button", name=click_button).first
                    button.click()
                    print(f"Waiting {wait_after_click}s for results...")

                    # 進行状況バーが消えるまで待機
                    page.wait_for_selector(
                        '[data-testid="stStatusWidget"]',
                        state="hidden",
                        timeout=wait_after_click * 1000,
                    )
                    time.sleep(2)  # 追加の安定化待機

                except Exception as e:
                    print(f"Button click warning: {e}")
                    # ボタンが見つからなくてもスクリーンショットは撮る
                    time.sleep(wait_after_click)

            # 最下部までスクロール（オプション）
            if scroll_to_bottom:
                print("Scrolling to bottom...")
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(1)

            # フルページスクリーンショット
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print("Capturing full-page screenshot...")
            page.screenshot(path=str(output_path), full_page=True)

            browser.close()
            print(f"Screenshot saved: {output_path}")
            return True
    except Exception as e:
        print(f"Screenshot failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Streamlit UI screenshot")
    parser.add_argument(
        "--url",
        default="http://localhost:8501",
        help="Streamlit app URL",
    )
    parser.add_argument(
        "--output",
        default="screenshots/streamlit_ui.png",
        help="Output screenshot path",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=3,
        help="Wait seconds after page load",
    )
    parser.add_argument(
        "--click-button",
        help="Button text to click (e.g., 'Run Today Signals')",
    )
    parser.add_argument(
        "--wait-after-click",
        type=int,
        default=15,
        help="Wait seconds after button click",
    )
    parser.add_argument(
        "--no-scroll",
        action="store_true",
        help="Do not scroll to bottom before screenshot",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / args.output

    success = capture_streamlit_screenshot(
        args.url,
        output_path,
        args.wait,
        args.click_button,
        args.wait_after_click,
        not args.no_scroll,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
