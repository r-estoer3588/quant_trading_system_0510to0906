"""Streamlit UIを起動してスクリーンショットを自動撮影する統合スクリプト"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import subprocess
import time

from playwright.async_api import async_playwright

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 設定
STREAMLIT_URL = "http://localhost:8501"
SCREENSHOT_DIR = Path("screenshots/progress_tracking")
SCREENSHOT_INTERVAL = 2.0
MAX_SCREENSHOTS = 500


async def capture_screenshots_after_ui_ready():
    """StreamlitUI起動待機後、スクリーンショット撮影開始"""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📸 スクリーンショット保存先: {SCREENSHOT_DIR.absolute()}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        logger.info("🌐 Chromiumブラウザ起動完了")

        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        logger.info(f"📄 Streamlit UIに接続中: {STREAMLIT_URL}")

        # UIの起動を待つ（最大30回リトライ = 60秒）
        for attempt in range(30):
            try:
                await page.goto(STREAMLIT_URL, wait_until="networkidle", timeout=5000)
                logger.info("✅ Streamlit UI接続完了")
                break
            except Exception as e:
                if attempt < 29:
                    logger.info(f"⏳ UI起動待機中... ({attempt + 1}/30)")
                    await asyncio.sleep(2)
                else:
                    logger.error(f"❌ UI接続タイムアウト: {e}")
                    await browser.close()
                    return

        # 当日シグナルタブに移動
        await asyncio.sleep(3)
        logger.info("🔍 「当日シグナル」タブを探しています...")

        tab_selectors = [
            'button:has-text("当日シグナル")',
            'div[data-testid="stTab"]:has-text("当日シグナル")',
            'button[role="tab"]:has-text("当日シグナル")',
        ]

        for selector in tab_selectors:
            try:
                tab = page.locator(selector).first
                if await tab.count() > 0:
                    await tab.click()
                    logger.info("✅ 「当日シグナル」タブクリック")
                    break
            except Exception:
                continue

        await asyncio.sleep(2)

        # 実行ボタンをクリック
        logger.info("🔍 「実行」ボタンを探しています...")

        button_selectors = [
            'button:has-text("実行")',
            'button:has-text("▶")',
            "div.stButton button",
        ]

        button_clicked = False
        for selector in button_selectors:
            try:
                btn = page.locator(selector).first
                await btn.wait_for(state="visible", timeout=5000)
                await btn.click()
                logger.info("✅ 実行ボタンクリック成功")
                button_clicked = True
                await asyncio.sleep(3)
                break
            except Exception:
                continue

        if not button_clicked:
            logger.error("❌ 実行ボタンが見つかりません")
            await browser.close()
            return

        # スクリーンショット撮影開始
        logger.info(f"📸 撮影開始（{SCREENSHOT_INTERVAL}秒間隔、最大{MAX_SCREENSHOTS}枚）")
        screenshot_count = 0

        try:
            while screenshot_count < MAX_SCREENSHOTS:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"progress_{ts}.png"
                filepath = SCREENSHOT_DIR / filename

                await page.screenshot(path=str(filepath), full_page=False)
                screenshot_count += 1

                if screenshot_count % 10 == 0:
                    logger.info(f"📸 撮影: {screenshot_count}枚")

                # 完了検出
                try:
                    selectors = [
                        "text=/✅.*完了/i",
                        "text=/完了しました/i",
                    ]
                    for sel in selectors:
                        if await page.locator(sel).count() > 0:
                            logger.info("✅ 完了検出 - 撮影終了")
                            for i in range(3):
                                await asyncio.sleep(1.0)
                                ts2 = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                fn = f"progress_{ts2}_final{i + 1}.png"
                                fp = SCREENSHOT_DIR / fn
                                await page.screenshot(path=str(fp), full_page=False)
                                screenshot_count += 1
                            raise StopIteration
                except StopIteration:
                    break
                except Exception:
                    pass

                await asyncio.sleep(SCREENSHOT_INTERVAL)

        except KeyboardInterrupt:
            logger.info("⚠️ ユーザー中断")
        finally:
            logger.info(f"📸 撮影終了: 合計{screenshot_count}枚")
            await browser.close()


def main():
    """メイン処理"""
    # Streamlit UIをバックグラウンドで起動
    logger.info("🚀 Streamlit UI起動中...")
    venv_python = Path("venv/Scripts/python.exe")

    streamlit_proc = subprocess.Popen(
        [str(venv_python), "-m", "streamlit", "run", "apps/app_today_signals.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    logger.info("⏳ UI起動待機中（10秒）...")
    time.sleep(10)

    # スクリーンショット撮影開始
    try:
        asyncio.run(capture_screenshots_after_ui_ready())
    except KeyboardInterrupt:
        logger.info("⚠️ ユーザー中断")
    finally:
        # Streamlitプロセス終了
        logger.info("🛑 Streamlit UI停止中...")
        streamlit_proc.terminate()
        try:
            streamlit_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamlit_proc.kill()
        logger.info("✅ 完了")


if __name__ == "__main__":
    main()
