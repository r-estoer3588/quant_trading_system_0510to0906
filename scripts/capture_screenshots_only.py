"""Streamlit UIのスクリーンショットを定期的に撮影するスクリプト（実行は手動）"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path

from playwright.async_api import async_playwright

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 設定
STREAMLIT_URL = "http://localhost:8501"
SCREENSHOT_DIR = Path("screenshots/progress_tracking")
SCREENSHOT_INTERVAL = 2.0  # 2秒ごとに撮影
MAX_DURATION = 600  # 最大10分間撮影


async def capture_screenshots():
    """Streamlit UIのスクリーンショットを定期的に撮影（実行は手動）"""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📸 スクリーンショット保存先: {SCREENSHOT_DIR.absolute()}")

    async with async_playwright() as p:
        # Chromiumブラウザを起動（ヘッドレスモード）
        browser = await p.chromium.launch(headless=True)
        logger.info("🌐 Chromiumブラウザ起動完了")

        # 新しいページを開く
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        logger.info(f"📄 Streamlit UIに接続中: {STREAMLIT_URL}")

        try:
            await page.goto(STREAMLIT_URL, wait_until="networkidle", timeout=30000)
            logger.info("✅ Streamlit UI接続完了")
        except Exception as e:
            logger.error(f"❌ Streamlit UIへの接続失敗: {e}")
            await browser.close()
            return

        # スクリーンショット撮影ループ
        screenshot_count = 0
        start_time = asyncio.get_event_loop().time()
        logger.info(
            f"📸 スクリーンショット撮影開始"
            f"（{SCREENSHOT_INTERVAL}秒間隔、最大{MAX_DURATION}秒）"
        )
        logger.info("👉 ブラウザで手動で「実行」ボタンをクリックしてください")

        try:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > MAX_DURATION:
                    logger.info(f"⏰ 最大撮影時間（{MAX_DURATION}秒）到達 - 終了")
                    break

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"progress_{timestamp}.png"
                filepath = SCREENSHOT_DIR / filename

                # スクリーンショット撮影
                await page.screenshot(path=str(filepath), full_page=False)
                screenshot_count += 1

                if screenshot_count % 30 == 0:
                    logger.info(
                        f"📸 撮影中: {screenshot_count}枚 "
                        f"（経過時間: {int(elapsed)}秒）"
                    )

                # 完了メッセージを検出したら終了
                try:
                    # Streamlitの成功メッセージを検出
                    completion_locators = [
                        page.locator("text=/✅.*完了/i"),
                        page.locator("text=/🎉.*完了/i"),
                        page.locator("text=/完了しました/i"),
                        page.locator('[data-testid="stSuccess"]'),
                    ]

                    for locator in completion_locators:
                        if await locator.count() > 0:
                            logger.info("✅ 完了メッセージ検出 - 撮影終了")
                            # 最後に5枚追加撮影
                            for i in range(5):
                                await asyncio.sleep(1.0)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                                    :-3
                                ]
                                filename = f"progress_{timestamp}_final{i+1}.png"
                                filepath = SCREENSHOT_DIR / filename
                                await page.screenshot(
                                    path=str(filepath), full_page=False
                                )
                                screenshot_count += 1
                            raise StopIteration()
                except StopIteration:
                    break
                except Exception:
                    pass

                await asyncio.sleep(SCREENSHOT_INTERVAL)

        except KeyboardInterrupt:
            logger.info("⚠️ ユーザーによる中断")
        except Exception as e:
            logger.error(f"❌ 撮影エラー: {e}")
        finally:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(
                f"📸 撮影終了: 合計{screenshot_count}枚 "
                f"（撮影時間: {int(elapsed)}秒）"
            )
            await browser.close()
            logger.info("🌐 ブラウザ終了")


if __name__ == "__main__":
    asyncio.run(capture_screenshots())
