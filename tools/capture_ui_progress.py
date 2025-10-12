"""
リアルタイムUI進捗スクリーンショットキャプチャツール

使い方:
    python tools/capture_ui_progress.py --interval 2 --output screenshots/progress_tracking
"""

import argparse
from datetime import datetime
from pathlib import Path
import time

from playwright.sync_api import sync_playwright


def capture_screenshots(
    url: str,
    interval: int,
    output_dir: Path,
    max_captures: int = 100,
    wait_timeout_sec: int = 30,
) -> None:
    """指定間隔でStreamlit UIのスクリーンショットを撮影。

    - url がまだ起動していない場合は wait_timeout_sec の範囲でリトライして待機する。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--force-dark-mode",
                "--blink-settings=forceDarkModeEnabled=true",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            color_scheme="dark",
        )
        page = context.new_page()
        # 最初の到達をリトライ
        start = time.time()
        while True:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=10_000)
                break
            except Exception:
                if time.time() - start > wait_timeout_sec:
                    raise
                time.sleep(1)

        print(f"📸 スクリーンショット監視開始: {url}")
        print(f"   保存先: {output_dir}")
        print(f"   間隔: {interval}秒")
        print("   Ctrl+C で停止\n")

        try:
            for i in range(max_captures):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = output_dir / f"progress_{timestamp}.png"

                # スクリーンショット撮影
                page.screenshot(path=str(filename), full_page=True)
                print(f"[{i+1:3d}] {timestamp} - 保存完了: {filename.name}")

                # 進捗バーの値を抽出して表示 (オプション)
                try:
                    # Streamlitの進捗バー要素を検索
                    progress_elements = page.locator('[data-testid="stProgress"]').all()
                    if progress_elements:
                        print(f"      進捗バー検出: {len(progress_elements)}個")
                except Exception:
                    pass

                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n✅ 監視停止 - 合計{i+1}枚撮影")
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streamlit UI進捗スクリーンショット監視"
    )
    parser.add_argument("--url", default="http://localhost:8501", help="Streamlit URL")
    parser.add_argument("--interval", type=int, default=2, help="撮影間隔(秒)")
    parser.add_argument(
        "--output", default="screenshots/progress_tracking", help="保存先ディレクトリ"
    )
    parser.add_argument("--max", type=int, default=100, help="最大撮影枚数")
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=30,
        help="起動待機の最大秒数 (URL到達リトライ)",
    )

    args = parser.parse_args()

    capture_screenshots(
        url=args.url,
        interval=args.interval,
        output_dir=Path(args.output),
        max_captures=args.max,
        wait_timeout_sec=args.wait_timeout,
    )


if __name__ == "__main__":
    main()
