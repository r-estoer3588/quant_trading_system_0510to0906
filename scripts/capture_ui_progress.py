"""Streamlit UIの進捗を自動撮影するスクリプト（Playwright）

改訂（2025-10-13）:
- 実行ボタンの正確なラベルに対応（「▶ 本日のシグナル実行」）
- 「当日シグナル」タブへの依存を解消（該当タブは存在しない）
- スクロール探索 + 複数セレクタでボタンクリックの堅牢性を向上
- ヘッドあり/ダークモード強制に対応、終了時のclose例外は握りつぶし
- 完了検知を厳格化（JSONLのsystem7完了 or DOMの「実行終了」系メッセージ）
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from playwright.async_api import Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# 設定（デフォルトURL。CLI --url または環境変数で上書き可能）
DEFAULT_STREAMLIT_URL = "http://localhost:8501"
SCREENSHOT_DIR = Path("screenshots/progress_tracking")
SCREENSHOT_INTERVAL = 2.0  # 秒
MAX_SCREENSHOTS = 500
# Candidate labels the run button may use (keeps backward compatibility).
# Prefer English 'Generate Signals' but also try common Japanese variants.
BUTTON_CANDIDATES = [
    "Generate Signals",
    "本日のシグナル実行",
    "▶ 本日のシグナル実行",
    "シグナル生成",
    "シグナル生成を開始",
]


async def _try_click_run_button(page: Page) -> bool:
    """『▶ 本日のシグナル実行』ボタンを探索してクリックする。

    可能な限り複数のセレクタで探索し、画面をスクロールしながら探す。
    見つからなければ False。
    """

    # Build selectors for each candidate label to improve robustness across
    # different language / icon variants.
    selectors = []
    for candidate in BUTTON_CANDIDATES:
        try:
            selectors.append(
                lambda c=candidate: page.get_by_role("button", name=re.compile(c))
            )
            selectors.append(lambda c=candidate: page.get_by_text(f"▶ {c}", exact=True))
            selectors.append(lambda c=candidate: page.get_by_text(c))
            selectors.append(
                lambda c=candidate: page.locator("button").filter(
                    has_text=re.compile(c)
                )
            )
            selectors.append(
                lambda c=candidate: page.locator("div.stButton button").filter(
                    has_text=re.compile(c)
                )
            )
        except Exception:
            continue

    # スクロールしながら最大 N 回探索
    max_scroll_steps = 8
    for step in range(max_scroll_steps):
        if step > 0:
            # 下方向にスクロール
            await page.evaluate(
                "window.scrollBy(0, Math.floor(window.innerHeight * 0.9));"
            )
            await asyncio.sleep(0.5)

        # 各スクロール位置で全セレクタを試す
        for make_locator in selectors:
            try:
                loc = make_locator()
                await loc.first.wait_for(state="visible", timeout=1500)
                await loc.first.scroll_into_view_if_needed()
                await asyncio.sleep(0.1)
                await loc.first.click()
                logger.info("✅ 『▶ 本日のシグナル実行』ボタンクリック成功")
                return True
            except PlaywrightTimeoutError:
                # 見つからなかった場合は次のセレクタで試す
                continue
            except Exception:
                # その他の例外は握りつぶして別セレクタで再試行
                continue

    # 先頭に戻って再試行
    await page.evaluate("window.scrollTo(0, 0);")
    await asyncio.sleep(0.3)
    for make_locator in selectors:
        try:
            loc = make_locator()
            await loc.first.wait_for(state="visible", timeout=1500)
            await loc.first.scroll_into_view_if_needed()
            await asyncio.sleep(0.1)
            await loc.first.click()
            logger.info("✅ 『▶ 本日のシグナル実行』ボタンクリック成功（2回目探索）")
            return True
        except Exception:
            continue

    return False


def _jsonl_has_system7_complete() -> bool:
    """logs/progress_today.jsonl に system7 の system_complete が現れたか判定。

    失敗は無視（False）し、最大200行だけを対象に軽量チェック。
    """
    try:
        jl = Path("logs/progress_today.jsonl")
        if not jl.exists():
            return False
        tail_lines = jl.read_text(encoding="utf-8", errors="ignore").splitlines()[-200:]
        for ln in reversed(tail_lines):
            if '"event_type": "system_complete"' in ln and '"system": "system7"' in ln:
                return True
        return False
    except Exception:
        return False


def _jsonl_has_pipeline_complete() -> bool:
    """logs/progress_today.jsonl に pipeline_complete が現れたか判定。"""
    try:
        jl = Path("logs/progress_today.jsonl")
        if not jl.exists():
            return False
        tail = jl.read_text(encoding="utf-8", errors="ignore").splitlines()[-200:]
        for ln in reversed(tail):
            if '"event_type": "pipeline_complete"' in ln:
                return True
        return False
    except Exception:
        return False


async def _dom_has_explicit_finish(page: Page) -> bool:
    """DOMに『実行終了』などの明示的な完了表示が出たかを判定。"""
    try:
        dom_completion = [
            'text="本日のシグナル 実行終了"',
            "text=/シグナル検出処理\\s*終了/i",
            "text=/実行\\s*終了/i",
            "text=/Engine.*実行終了/i",
        ]
        for selector in dom_completion:
            el = page.locator(selector)
            if await el.count() > 0:
                logger.info(f"✅ 画面で実行終了を検出（{selector}）")
                return True
    except Exception:
        pass
    return False


async def capture_screenshots() -> None:
    """Streamlit UIの進捗バーを自動撮影"""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📸 スクリーンショット保存先: {SCREENSHOT_DIR.absolute()}")

    # ヘッドあり/なしの決定（優先度: CLI --headed > env PLAYWRIGHT_HEADFUL > env PLAYWRIGHT_HEADLESS）
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--slowmo", type=int, default=0)
    parser.add_argument("--dark", dest="dark", action="store_true")
    parser.add_argument("--light", dest="dark", action="store_false")
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("STREAMLIT_URL", DEFAULT_STREAMLIT_URL),
    )
    parser.set_defaults(dark=True)
    try:
        args, _ = parser.parse_known_args()
    except SystemExit:
        # 解析失敗時はデフォルト値
        args = argparse.Namespace(
            headed=False,
            slowmo=0,
            dark=True,
            url=os.environ.get("STREAMLIT_URL", DEFAULT_STREAMLIT_URL),
        )

    env_headful = os.environ.get("PLAYWRIGHT_HEADFUL", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    env_headless = os.environ.get("PLAYWRIGHT_HEADLESS", "").lower()
    env_dark = os.environ.get("PLAYWRIGHT_DARK")

    if args.headed or env_headful:
        headless = False
    else:
        headless = True if env_headless in ("", "1", "true", "yes") else False

    dark = args.dark
    if env_dark is not None:
        dark = env_dark.lower() in ("1", "true", "yes")

    async with async_playwright() as p:
        # Chromiumブラウザを起動
        launch_args = ["--force-color-profile=srgb"]
        if dark:
            # ダークモードを強制（ヘッドありで特に有効）
            launch_args += [
                "--force-dark-mode",
                "--enable-features=WebContentsForceDark",
                "--blink-settings=forceDarkModeEnabled=true",
            ]
        browser = await p.chromium.launch(
            headless=headless, slow_mo=args.slowmo, args=launch_args
        )
        logger.info(
            "🌐 Chromiumブラウザ起動完了 (%s) slowMo=%sms, color-scheme=%s",
            "headed" if not headless else "headless",
            args.slowmo,
            "dark" if dark else "light",
        )

        # 新しいページを開く
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            color_scheme="dark" if dark else "light",
            locale="ja-JP",
        )
        page = await context.new_page()
        await page.emulate_media(color_scheme="dark" if dark else "light")
        page.set_default_timeout(15000)
        target_url = args.url or DEFAULT_STREAMLIT_URL
        logger.info(f"📄 Streamlit UIに接続中: {target_url}")

        try:
            await page.goto(target_url, wait_until="networkidle", timeout=30000)
            logger.info("✅ Streamlit UI接続完了")
        except Exception as e:
            logger.warning(f"⚠️ UI接続に失敗しました（{e}）")
            # 8501→8502 の簡易フォールバック（localhostのみ）
            try:
                if target_url.startswith("http://localhost:8501"):
                    fb = target_url.replace(":8501", ":8502")
                    logger.info(f"🔁 フォールバック接続を試行: {fb}")
                    await page.goto(fb, wait_until="networkidle", timeout=30000)
                    logger.info("✅ Streamlit UI接続完了（フォールバック）")
                    target_url = fb
                else:
                    raise
            except Exception as e2:
                logger.error(f"❌ Streamlit UIへの接続失敗: {e2}")
                try:
                    await browser.close()
                except Exception:
                    pass
                return

        # 初期レンダリング待機
        await asyncio.sleep(1.5)

        # 実行ボタンをクリック（スクロール探索込み）
        try:
            logger.info(
                "🔍 Looking for run button: 'Generate Signals' / '本日のシグナル実行' ..."
            )
            clicked = await _try_click_run_button(page)
            if not clicked:
                logger.warning("⚠️ 実行ボタンが見つかりません - 手動で実行してください")
            else:
                logger.info("🚀 シグナル抽出を開始しました")
                await asyncio.sleep(3)  # 実行開始待機
        except Exception as e:
            logger.warning(f"⚠️ 実行ボタンクリックエラー: {e}")

        # スクリーンショット撮影ループ
        screenshot_count = 0
        logger.info(
            f"📸 スクリーンショット撮影開始（{SCREENSHOT_INTERVAL}秒間隔、最大{MAX_SCREENSHOTS}枚）"
        )

        try:
            while screenshot_count < MAX_SCREENSHOTS:
                # 1枚撮影
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"progress_{timestamp}.png"
                filepath = SCREENSHOT_DIR / filename
                await page.screenshot(path=str(filepath), full_page=False)
                screenshot_count += 1

                if screenshot_count % 10 == 0:
                    logger.info(f"📸 撮影完了: {screenshot_count}枚")

                # 完了判定（厳格版）
                try:
                    # 0) JSONL: pipeline_complete（全体完了）
                    if _jsonl_has_pipeline_complete():
                        logger.info(
                            "✅ JSONLでpipeline_completeを検出 - 追い撮りして終了"
                        )
                        for i in range(5):
                            await asyncio.sleep(0.8)
                            ts2 = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            fn2 = f"progress_{ts2}_final{i + 1}.png"
                            await page.screenshot(
                                path=str(SCREENSHOT_DIR / fn2), full_page=False
                            )
                            screenshot_count += 1
                        return

                    # 1) JSONL: system7 の system_complete を検出
                    if _jsonl_has_system7_complete():
                        logger.info("✅ JSONLでsystem7完了を検出 - 追い撮りして終了")
                        for i in range(5):
                            await asyncio.sleep(0.8)
                            ts2 = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            fn2 = f"progress_{ts2}_final{i + 1}.png"
                            await page.screenshot(
                                path=str(SCREENSHOT_DIR / fn2), full_page=False
                            )
                            screenshot_count += 1
                        return

                    # 2) DOM: 明示的な『実行終了』表示
                    if await _dom_has_explicit_finish(page):
                        logger.info("✅ 画面の『実行終了』を検出 - 追い撮りして終了")
                        for i in range(5):
                            await asyncio.sleep(0.8)
                            ts2 = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            fn2 = f"progress_{ts2}_final{i + 1}.png"
                            await page.screenshot(
                                path=str(SCREENSHOT_DIR / fn2), full_page=False
                            )
                            screenshot_count += 1
                        return
                except Exception:
                    # 完了検出に失敗しても継続
                    pass

                await asyncio.sleep(SCREENSHOT_INTERVAL)

        except KeyboardInterrupt:
            logger.info("⚠️ ユーザーによる中断")
        except Exception as e:
            logger.error(f"❌ 撮影エラー: {e}")
        finally:
            logger.info(f"📸 撮影終了: 合計{screenshot_count}枚")
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                # 一部環境で close 時にドライバー切断例外が出るため握りつぶす
                pass
            logger.info("🌐 ブラウザ終了")


if __name__ == "__main__":
    try:
        asyncio.run(capture_screenshots())
    except KeyboardInterrupt:
        logger.info("⏹️ ユーザー中断: 終了します")
    except asyncio.CancelledError:
        logger.info("⏹️ キャンセル要求を受信: 終了します")
    except Exception as e:
        logger.error(f"❌ 致命的エラー: {e}")
