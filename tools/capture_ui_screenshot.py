"""Streamlit UI のスクリーンショットを自動取得するツール。

Playwright を使用して、実行中の Streamlit アプリのスクリーンショットを保存します。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Literal

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
except ImportError:  # pragma: no cover - handled later when playwright import fails
    PlaywrightTimeoutError = Exception


def capture_streamlit_screenshot(
    url: str,
    output_path: Path,
    wait_seconds: int = 3,
    click_button: str | None = None,
    wait_after_click: int = 15,
    scroll_to_bottom: bool = True,
    headless: bool = True,
    wait_for_user: bool = False,
    color_scheme: Literal["dark", "light", "no-preference"] = "dark",
    # New: robust post-click waits for results screen
    wait_text: str | None = None,
    wait_selector: str | None = None,
    wait_results: bool = False,
    # New: robust completion waits
    wait_jsonl: bool = False,
    jsonl_path: Path | None = None,
    wait_progress_pct: int | None = None,
) -> bool:
    """Streamlit UI のスクリーンショットを取得。

    Args:
        url: Streamlit アプリの URL (例: http://localhost:8501)
        output_path: 保存先パス
        wait_seconds: ページ読み込み待機時間（秒）
        click_button: クリックするボタンのテキスト（例: "Run Today Signals"）
        wait_after_click: ボタンクリック後の待機時間（秒）
        scroll_to_bottom: 最下部までスクロールするか（デフォルト: True）
        headless: ヘッドレスモード（False でブラウザ表示）
        wait_for_user: ユーザーが Enter を押すまで待機
        color_scheme: ブラウザのカラースキーム（'dark' または 'light'）

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
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},  # 標準的なビューポート
                color_scheme=color_scheme,
            )
            page = context.new_page()

            print(f"Loading: {url}")
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(wait_seconds)

            def locate_system5():
                """System5 の要素を探し、中央表示とバウンディングボックスを返す。"""

                candidate_texts = [
                    "System5",
                    "System 5",
                    "システム5",
                    "System5 ",  # 末尾にスペースが付くケース
                    "System 5 ",
                ]

                for text in candidate_texts:
                    locator = page.get_by_text(text, exact=False).first
                    try:
                        locator.wait_for(state="visible", timeout=4000)
                        locator.evaluate(
                            "el => el.scrollIntoView({behavior: 'instant', block: 'center', inline: 'center'})"
                        )
                        time.sleep(0.2)
                        box = locator.bounding_box()
                        if box:
                            print(
                                "System5 element bounding box: "
                                f"x={box['x']:.1f}, y={box['y']:.1f}, "
                                f"w={box['width']:.1f}, h={box['height']:.1f}"
                            )
                            return locator, box
                    except PlaywrightTimeoutError:
                        continue
                    except Exception as locator_error:  # pragma: no cover
                        print(f"System5 locator warning: {locator_error}")
                        continue

                print("Warning: System5 element not located; layout may differ.")
                return None, None

            # Streamlit の CSS とコンポーネントの読み込みを待機
            print("Waiting for Streamlit to fully render...")
            try:
                # 1. Streamlit のメインコンテナが表示されるまで待機
                page.wait_for_selector(
                    '[data-testid="stAppViewContainer"]', timeout=10000
                )

                # 2. すべてのリソースとネットワーク読み込み完了を待機
                page.wait_for_load_state("load")
                page.wait_for_load_state("networkidle")

                # 3. Streamlit の columns 要素がレンダリングされるまで待機（堅牢化）
                print("Waiting for Streamlit columns to render...")

                def wait_for_columns_robust(total_timeout_ms: int = 15000) -> None:
                    """Streamlit columns 検出の堅牢化。

                    - 複数セレクタを順に試す（Streamlit バージョン差分吸収）
                    - エクスポネンシャルバックオフでリトライ
                    - いずれかのセレクタで要素が見つかれば成功とみなす
                    """

                    candidate_selectors = [
                        '[data-testid="column"]',
                        '[data-testid="stVerticalBlock"]',
                        '[data-testid="stHorizontalBlock"]',
                        '[data-testid="stElementContainer"]',
                        "div.block-container",
                    ]

                    # リトライ: 2s -> 4s -> 8s （合計 14s 目安）
                    time_buckets = [
                        2000,
                        4000,
                        max(1000, total_timeout_ms - 6000),
                    ]
                    started_at = time.time()

                    for bucket in time_buckets:
                        remaining = max(
                            0,
                            total_timeout_ms - int((time.time() - started_at) * 1000),
                        )
                        if remaining <= 0:
                            break
                        per_selector_timeout = min(bucket, remaining)

                        for sel in candidate_selectors:
                            try:
                                # いずれかのセレクタで 1 要素でも見つかれば OK
                                locator = page.locator(sel)
                                count = locator.count()
                                if count and count > 0:
                                    # 既に存在するなら可視化を待機して抜ける
                                    locator.first.wait_for(
                                        state="visible",
                                        timeout=per_selector_timeout,
                                    )
                                    print(f"Columns detected via selector: {sel}")
                                    return
                                # 存在しない場合はこのセレクタで待つ
                                page.wait_for_selector(
                                    sel,
                                    timeout=per_selector_timeout,
                                )
                                print(f"Columns detected via selector: {sel}")
                                return
                            except PlaywrightTimeoutError:
                                continue
                            except Exception as wait_err:  # pragma: no cover
                                print(f"Column wait warning ({sel}): {wait_err}")
                                continue

                    # 最後まで見つからなければ警告
                    print(
                        "Warning: Could not detect Streamlit columns in time; proceeding with fallback waits."
                    )

                wait_for_columns_robust(total_timeout_ms=15000)

                # 4. Streamlit の内部 JavaScript とレイアウト計算完了
                # Streamlit は React ベースで非同期レンダリングするため
                # 十分な時間を確保
                time.sleep(5)

            except Exception as e:
                print(f"Warning: Streamlit component detection failed: {e}")
                # フォールバック：さらに長く待機
                time.sleep(8)

            # ボタンクリック（オプション）
            if click_button:
                print(f"Clicking button: '{click_button}'")
                try:
                    # クリック（ロール指定）
                    try:
                        button = page.get_by_role("button", name=click_button).first
                        button.click()
                    except Exception as e_role:
                        # フォールバック: テキスト検索でクリック
                        print(
                            "Primary click failed ("
                            + str(e_role)
                            + "); trying text-based fallback..."
                        )
                        fallback = page.get_by_text(click_button, exact=False).first
                        fallback.wait_for(state="visible", timeout=8000)
                        fallback.click()

                    if wait_for_user:
                        print("\n" + "=" * 60)
                        print("ボタンをクリックしました。")
                        print("処理が完了したら Enter キーを押してください...")
                        print("=" * 60)
                        input()

                        # ユーザー入力後も Streamlit の再レンダリングを待機
                        print("Waiting for Streamlit to re-render...")
                        time.sleep(3)
                        page.wait_for_load_state("networkidle", timeout=10000)
                        time.sleep(2)
                    else:
                        print(f"Waiting {wait_after_click}s for results...")
                        # 進行状況バーが消えるまで待機
                        page.wait_for_selector(
                            '[data-testid="stStatusWidget"]',
                            state="hidden",
                            timeout=wait_after_click * 1000,
                        )
                        # Streamlit の結果表示後の再レンダリング完了を待機
                        time.sleep(3)
                        page.wait_for_load_state("networkidle", timeout=10000)
                        time.sleep(2)
                except Exception as e:
                    print(f"Button click warning: {e}")
                    # ボタンが見つからなくてもスクリーンショットは撮る
                    if not wait_for_user:
                        time.sleep(wait_after_click)

            # クリック後の「結果画面」をより確実に待機（任意指定 or 既定の特徴語）
            try:
                # 合計待機時間（ミリ秒）をクリック後待機時間から推定、下限を設定
                total_wait_ms = max(10000, int(wait_after_click * 1000))
                deadline = time.time() + (total_wait_ms / 1000.0)

                def _remaining_ms() -> int:
                    return max(0, int((deadline - time.time()) * 1000))

                def _wait_text(txt: str) -> bool:
                    try:
                        loc = page.get_by_text(txt, exact=False).first
                        loc.wait_for(
                            state="visible", timeout=max(1000, _remaining_ms())
                        )
                        return True
                    except Exception:
                        return False

                def _wait_selector(sel: str) -> bool:
                    try:
                        page.wait_for_selector(
                            sel, state="visible", timeout=max(1000, _remaining_ms())
                        )
                        return True
                    except Exception:
                        return False

                # JSONL/pipeline 完了検出
                def _jsonl_has_pipeline_complete_since(
                    path: Path, since_epoch: float
                ) -> bool:
                    try:
                        if not path.exists():
                            return False
                        lines = path.read_text(
                            encoding="utf-8", errors="ignore"
                        ).splitlines()
                        for ln in reversed(
                            lines[-500:]
                        ):  # 末尾側（最大500行）だけを見る
                            if "pipeline_complete" not in ln:
                                continue
                            # できればtimestampを確認
                            try:
                                obj = json.loads(ln)
                                ts = obj.get("timestamp")
                                # 簡易比較: timestampがあれば現在時刻との差で新しいか判定、なければヒット扱い
                                if ts:
                                    # タイムスタンプ厳密変換が難しい環境もあるため、ファイルの末尾側に出たら新しいとみなす
                                    return True
                                return True
                            except Exception:
                                return True
                        return False
                    except Exception:
                        return False

                def _dom_progress_pct() -> int | None:
                    try:
                        value = page.evaluate(
                            r"""
(() => {
  const nodes = Array.from(document.querySelectorAll('h1,h2,h3,div,p,span'));
  for (const el of nodes) {
    const t = (el.innerText || '').trim();
    const m = t.match(/進捗\s*(\d+)%/);
    if (m) return parseInt(m[1], 10);
  }
  return null;
})()
"""
                        )
                        if value is None:
                            return None
                        try:
                            return int(value)
                        except Exception:
                            return None
                    except Exception:
                        return None

                waited = False
                # 明示指定があれば優先
                if wait_text:
                    print(f"Waiting for text to appear: '{wait_text}'")
                    waited = _wait_text(wait_text)
                if not waited and wait_selector:
                    print(f"Waiting for selector to appear: {wait_selector}")
                    waited = _wait_selector(wait_selector)

                # 既定の「結果画面」候補を順に待機
                if not waited and wait_results:
                    print("Waiting for results screen markers...")
                    candidates_text = [
                        "本日のシグナル実行完了",
                        "最終選定銘柄",
                        "本日のシグナルはありません",
                        "総経過時間:",
                    ]
                    candidates_buttons = [
                        "最終CSVをダウンロード",
                    ]
                    # 1) 見出しや情報文
                    for t in candidates_text:
                        if _wait_text(t):
                            waited = True
                            break
                    # 2) ダウンロードボタン（ロール）
                    if not waited:
                        for name in candidates_buttons:
                            try:
                                loc = page.get_by_role("button", name=name).first
                                loc.wait_for(
                                    state="visible", timeout=max(1000, _remaining_ms())
                                )
                                waited = True
                                break
                            except Exception:
                                continue
                if not waited and (wait_jsonl or (wait_progress_pct is not None)):
                    print("Polling JSONL/progress until completion criteria match...")
                    # ポーリング: 1秒間隔
                    started = time.time()
                    jl_path = (
                        jsonl_path
                        if jsonl_path
                        else (
                            Path(__file__).resolve().parents[1]
                            / "logs"
                            / "progress_today.jsonl"
                        )
                    )
                    while time.time() < deadline:
                        if wait_jsonl and _jsonl_has_pipeline_complete_since(
                            jl_path, since_epoch=started
                        ):
                            print("Detected pipeline_complete in JSONL.")
                            waited = True
                            break
                        if wait_progress_pct is not None:
                            pct = _dom_progress_pct()
                            if isinstance(pct, int) and pct >= int(wait_progress_pct):
                                print(
                                    f"Detected progress >= {wait_progress_pct}% (now {pct}%)."
                                )
                                waited = True
                                break
                        # 軽負荷スリープ
                        time.sleep(1.0)

                if waited:
                    print("Results screen detected.")
                else:
                    print("Results markers not detected in time — proceeding anyway.")
            except Exception as _res_e:
                print(f"Results wait warning: {_res_e}")

            # 最下部までスクロール（オプション）
            if scroll_to_bottom:
                print("Scrolling to bottom...")
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(1)

            # コンテンツの実際のサイズを取得
            content_width = int(page.evaluate("document.body.scrollWidth") or 0)
            content_height = int(page.evaluate("document.body.scrollHeight") or 0)
            vp = page.viewport_size or {"width": 1920, "height": 1080}
            print(
                f"Content size: {content_width}x{content_height} px (viewport: {vp['width']}x{vp['height']})"
            )

            # 横方向の見切れ回避: コンテンツ幅がビューポートを超える場合は一時的に拡張
            # 安全上限（ユーザー画面に過度にならない範囲）を設定
            MAX_VIEWPORT_WIDTH = 2300
            if content_width and vp and content_width > int(vp.get("width", 1920)):
                new_width = min(int(content_width), MAX_VIEWPORT_WIDTH)
                if new_width > vp["width"]:
                    print(
                        f"Expanding viewport width: {vp['width']} -> {new_width} to avoid horizontal cutoff"
                    )
                    page.set_viewport_size(
                        {
                            "width": new_width,
                            "height": vp["height"],
                        }
                    )
                    # レイアウト再計算待機
                    time.sleep(0.5)
                    # 再度サイズ取得（高さが変わる可能性がある）
                    content_height = int(
                        page.evaluate("document.body.scrollHeight") or content_height
                    )
                    vp = page.viewport_size or vp
                    print(
                        f"Adjusted content: {content_width}x{content_height} px "
                        f"(viewport: {vp['width']}x{vp['height']})"
                    )

            # System5 の位置を記録し、見切れを検出
            locator, box = locate_system5()
            if box:
                vp = page.viewport_size or {"width": 1920, "height": 1080}
                viewport_width = int(vp.get("width", 1920))
                right_edge = box["x"] + box["width"]

                if right_edge > viewport_width and viewport_width < MAX_VIEWPORT_WIDTH:
                    new_width = min(int(right_edge) + 40, MAX_VIEWPORT_WIDTH)
                    if new_width > viewport_width:
                        print(
                            f"Expanding viewport for System5: {viewport_width} -> {new_width}"
                        )
                        page.set_viewport_size(
                            {
                                "width": new_width,
                                "height": vp["height"],
                            }
                        )
                        time.sleep(0.5)
                        locator, box = locate_system5()
                        vp = page.viewport_size or vp
                        viewport_width = int(vp.get("width", viewport_width))

                if box:
                    print(
                        "Final System5 bounding box: "
                        f"x={box['x']:.1f}, y={box['y']:.1f}, "
                        f"w={box['width']:.1f}, h={box['height']:.1f} "
                        f"(viewport width {viewport_width})"
                    )

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
        "--wait-text",
        help="Wait until text appears after click (partial match)",
    )
    parser.add_argument(
        "--wait-selector",
        help="Wait until CSS selector appears after click",
    )
    parser.add_argument(
        "--wait-results",
        action="store_true",
        help="Wait for common results markers (最終選定銘柄/本日のシグナルはありません/総経過時間/最終CSVボタン)",
    )
    parser.add_argument(
        "--wait-jsonl",
        action="store_true",
        help="Wait until logs/progress_today.jsonl contains pipeline_complete",
    )
    parser.add_argument(
        "--jsonl-path",
        help="Path to progress_today.jsonl (default: logs/progress_today.jsonl)",
    )
    parser.add_argument(
        "--wait-progress-pct",
        type=int,
        help="Wait until UI header shows progress >= given percent (e.g., 85)",
    )
    parser.add_argument(
        "--no-scroll",
        action="store_true",
        help="Do not scroll to bottom before screenshot",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show browser window (headful mode for debugging)",
    )
    parser.add_argument(
        "--wait-for-user",
        action="store_true",
        help="Wait for user to press Enter before taking screenshot",
    )
    parser.add_argument(
        "--light-mode",
        action="store_true",
        help="Use light color scheme (default is dark)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / args.output

    jl_path = (
        Path(args.jsonl_path).resolve() if getattr(args, "jsonl_path", None) else None
    )
    success = capture_streamlit_screenshot(
        args.url,
        output_path,
        args.wait,
        args.click_button,
        args.wait_after_click,
        not args.no_scroll,
        headless=not args.show_browser,
        wait_for_user=args.wait_for_user,
        color_scheme="light" if args.light_mode else "dark",
        wait_text=args.wait_text,
        wait_selector=args.wait_selector,
        wait_results=bool(args.wait_results),
        wait_jsonl=bool(args.wait_jsonl),
        jsonl_path=jl_path,
        wait_progress_pct=args.wait_progress_pct,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
