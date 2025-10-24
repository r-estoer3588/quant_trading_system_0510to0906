"""
UI進捗スクリーンショット自動解析ツール

Playwright + OCRで以下を抽出:
- 進捗バーの%値
- 実行中システム名
- 候補数などメトリクス
- UI要素の位置・サイズ

Usage:
    python tools/analyze_ui_screenshots.py --input screenshots/progress_tracking --output analysis_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image


def extract_timestamp_from_filename(filename: str) -> str | None:
    """
    ファイル名からタイムスタンプを抽出

    Example: progress_20251013_065209_856.png -> "2025-10-13 06:52:09.856"
    """
    match = re.match(r"progress_(\d{8})_(\d{6})_(\d{3})\.png", filename)
    if match:
        date_str = match.group(1)  # 20251013
        time_str = match.group(2)  # 065209
        ms_str = match.group(3)  # 856

        # Format: YYYY-MM-DD HH:MM:SS.mmm
        formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}.{ms_str}"
        return formatted
    return None


def extract_progress_bar_percentage(img: Image.Image) -> float | None:
    """
    画像から進捗バーの%を抽出（色解析ベース）

    Streamlitの進捗バーは緑色の塗りつぶし領域で表現される。
    画像の特定領域（進捗バー位置）の緑ピクセル比率から%を推定。
    """
    try:
        width, height = img.size

        # Streamlit進捗バーの推定位置（画像上部、中央付近）
        # 実際のUIレイアウトに応じて調整が必要
        progress_bar_region = (
            int(width * 0.1),  # left
            int(height * 0.05),  # top
            int(width * 0.9),  # right
            int(height * 0.15),  # bottom
        )

        cropped = img.crop(progress_bar_region)

        # 緑色ピクセル（進捗バー）をカウント
        # Streamlitの進捗バーは通常 RGB(0, 200-255, 0) 付近
        green_pixels = 0
        total_pixels = 0

        for x in range(cropped.width):
            for y in range(cropped.height):
                pixel = cropped.getpixel((x, y))
                if isinstance(pixel, tuple) and len(pixel) >= 3:
                    r, g, b = pixel[:3]
                    total_pixels += 1

                    # 緑色判定（Streamlit進捗バー特有の色）
                    if g > 150 and r < 100 and b < 100:
                        green_pixels += 1

        if total_pixels > 0:
            percentage = (green_pixels / total_pixels) * 100
            return round(percentage, 1)

        return None

    except Exception as e:
        print(f"⚠️ 進捗バー抽出エラー: {e}")
        return None


def extract_text_regions(img: Image.Image) -> dict[str, str]:
    """
    OCRを使ってテキスト領域を抽出（簡易版）

    pytesseractが利用可能な場合はOCRを実行。
    未インストールの場合はスキップ。
    """
    try:
        import pytesseract

        # 画像全体からテキスト抽出
        text = pytesseract.image_to_string(img, lang="eng+jpn")

        # システム名を抽出（System1, System2, ...）
        system_match = re.search(r"System\s*[1-7]", text, re.IGNORECASE)
        system_name = system_match.group(0) if system_match else None

        # 候補数を抽出（"10 件", "candidates: 10" など）
        candidates_match = re.search(r"(\d+)\s*(件|candidates)", text, re.IGNORECASE)
        candidates = candidates_match.group(1) if candidates_match else None

        return {
            "system_name": system_name,
            "candidates": candidates,
            "raw_text": text[:200],  # 最初の200文字のみ保存
        }

    except ImportError:
        # pytesseract未インストール時はスキップ
        return {
            "system_name": None,
            "candidates": None,
            "raw_text": "(OCR unavailable - pytesseract not installed)",
        }
    except Exception as e:
        return {
            "system_name": None,
            "candidates": None,
            "raw_text": f"(OCR error: {e})",
        }


def analyze_screenshot(image_path: Path) -> dict[str, Any]:
    """スクリーンショット1枚を解析"""
    try:
        img = Image.open(image_path)
        width, height = img.size

        # タイムスタンプ抽出
        timestamp = extract_timestamp_from_filename(image_path.name)

        # 進捗バー%抽出
        progress_percentage = extract_progress_bar_percentage(img)

        # テキスト領域抽出（OCR）
        text_data = extract_text_regions(img)

        result = {
            "file": image_path.name,
            "timestamp": timestamp,
            "image_size": f"{width}x{height}",
            "progress_percentage": progress_percentage,
            "system_name": text_data.get("system_name"),
            "candidates": text_data.get("candidates"),
            "ocr_text_sample": text_data.get("raw_text", "")[:100],
        }

        print(
            f"✅ {image_path.name}: {progress_percentage}% | {text_data.get('system_name', '?')}"
        )

        return result

    except Exception as e:
        print(f"❌ {image_path.name}: {e}")
        return {
            "file": image_path.name,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="UI進捗スクショ自動解析")
    parser.add_argument(
        "--input",
        type=str,
        default="screenshots/progress_tracking",
        help="スクショディレクトリ",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="screenshots/progress_tracking/analysis_results.json",
        help="解析結果JSON出力先",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="解析枚数上限（テスト用）",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_file = Path(args.output)

    if not input_dir.exists():
        print(f"❌ 入力ディレクトリが見つかりません: {input_dir}")
        return

    # スクショファイル一覧取得
    screenshot_files = sorted(input_dir.glob("progress_*.png"))

    if args.limit:
        screenshot_files = screenshot_files[: args.limit]

    print(f"🔍 解析開始: {len(screenshot_files)} 枚")
    print(f"📂 入力: {input_dir}")
    print(f"📄 出力: {output_file}")
    print("")

    # 解析実行
    results = []
    for img_path in screenshot_files:
        result = analyze_screenshot(img_path)
        results.append(result)

    # JSON出力
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("")
    print(f"✅ 解析完了: {len(results)} 枚")
    print(f"📊 結果: {output_file}")

    # サマリー表示
    valid_progress = [
        r["progress_percentage"]
        for r in results
        if r.get("progress_percentage") is not None
    ]
    if valid_progress:
        print(
            f"📈 進捗バー範囲: {min(valid_progress):.1f}% ～ {max(valid_progress):.1f}%"
        )


if __name__ == "__main__":
    main()
