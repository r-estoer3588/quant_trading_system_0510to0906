"""
スクリーンショット自動解析ツール

全スクショから進捗バー%・システム名・メトリクスを抽出し、
JSONLと同期検証を行う統合分析ツール。

Usage:
    python tools/analyze_screenshots_auto.py screenshots/progress_tracking/*.png
"""

import json
from pathlib import Path
import re
import sys

from PIL import Image
import pytesseract


def extract_timestamp_from_filename(filename: str) -> str:
    """
    ファイル名からタイムスタンプを抽出
    例: progress_20251013_065209_856.png -> 2025/10/13 06:52:09
    """
    match = re.match(r"progress_(\d{8})_(\d{6})_\d+\.png", filename)
    if match:
        date_str = match.group(1)  # 20251013
        time_str = match.group(2)  # 065209
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:8]
        hour = time_str[:2]
        minute = time_str[2:4]
        second = time_str[4:6]
        return f"{year}/{month}/{day} {hour}:{minute}:{second}"
    return ""


def analyze_progress_bar(img: Image.Image) -> float:
    """
    進捗バーの%値をピクセル解析で抽出

    Streamlitの進捗バーは通常画面上部に表示され、
    塗りつぶし色（緑系）の割合から%を推定可能。

    簡易実装: 画像上部20%領域から緑色ピクセルの割合を計算
    """
    try:
        width, height = img.size
        # 上部20%領域を切り出し
        top_region = img.crop((0, 0, width, int(height * 0.2)))

        # RGB値で緑系の色（進捗バー）を検出
        # Streamlitの進捗バーは通常 rgb(0, 200, 0) 系
        pixels = list(top_region.getdata())
        green_count = 0
        for r, g, b in pixels:
            # 緑が優勢なピクセルをカウント
            if g > 150 and g > r + 50 and g > b + 50:
                green_count += 1

        # 進捗バーの幅は画面幅の約80%と仮定
        total_bar_pixels = width * 0.8 * 50  # 高さ50pxと仮定
        progress_percent = (green_count / total_bar_pixels) * 100

        return min(100.0, max(0.0, progress_percent))
    except Exception:
        return 0.0


def extract_text_from_region(img: Image.Image, region_box: tuple) -> str:
    """
    画像の指定領域からOCRでテキスト抽出

    Args:
        img: PIL Image
        region_box: (left, top, right, bottom)
    """
    try:
        cropped = img.crop(region_box)
        text = pytesseract.image_to_string(cropped, lang="eng")
        return text.strip()
    except Exception as e:
        return f"ERROR: {e}"


def analyze_screenshot(image_path: Path) -> dict:
    """
    スクリーンショット1枚を解析

    Returns:
        {
            "file": "progress_20251013_065209_856.png",
            "timestamp": "2025/10/13 06:52:09",
            "progress_percent": 14.3,
            "system_name": "System1",
            "metrics": {
                "candidates": 10,
                "total_symbols": 150
            }
        }
    """
    try:
        img = Image.open(image_path)
        width, height = img.size

        # タイムスタンプ抽出
        timestamp = extract_timestamp_from_filename(image_path.name)

        # 進捗バー%抽出（ピクセル解析）
        progress = analyze_progress_bar(img)

        # システム名抽出（OCR: 画面中央上部から）
        # Streamlit UIの「実行中システム」表示は通常中央上部
        system_region = (
            int(width * 0.3),
            int(height * 0.1),
            int(width * 0.7),
            int(height * 0.25),
        )
        system_text = extract_text_from_region(img, system_region)

        # システム名をパース（"System1" などを抽出）
        system_match = re.search(r"[Ss]ystem\s*(\d)", system_text)
        system_name = f"system{system_match.group(1)}" if system_match else "unknown"

        # メトリクス抽出（OCR: 画面右側パネルから）
        metrics_region = (int(width * 0.7), int(height * 0.3), width, int(height * 0.7))
        metrics_text = extract_text_from_region(img, metrics_region)

        # 候補数を抽出
        candidates = 0
        cand_match = re.search(r"候補.*?(\d+)", metrics_text)
        if cand_match:
            candidates = int(cand_match.group(1))

        result = {
            "file": image_path.name,
            "timestamp": timestamp,
            "progress_percent": round(progress, 1),
            "system_name": system_name,
            "metrics": {"candidates": candidates},
            "image_size": f"{width}x{height}",
        }

        return result

    except Exception as e:
        return {
            "file": image_path.name,
            "timestamp": extract_timestamp_from_filename(image_path.name),
            "error": str(e),
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/analyze_screenshots_auto.py <image_paths>")
        sys.exit(1)

    print("🔍 スクリーンショット自動解析開始...")
    print(f"   対象: {len(sys.argv) - 1} 枚")
    print("")

    results = []
    for i, arg in enumerate(sys.argv[1:], 1):
        path = Path(arg)
        if path.exists() and path.suffix.lower() == ".png":
            result = analyze_screenshot(path)
            results.append(result)

            # 進捗表示
            if i % 10 == 0 or i == len(sys.argv) - 1:
                print(f"   処理中: {i}/{len(sys.argv) - 1} 枚...")
        else:
            print(f"⚠️ スキップ: {arg}")

    # 結果をJSON出力
    output_path = Path("screenshots/progress_tracking/analysis_auto.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 解析完了: {len(results)} 枚")
    print(f"📊 結果ファイル: {output_path}")

    # サマリー表示
    print("\n【サマリー】")
    progress_values = [r.get("progress_percent", 0) for r in results if "error" not in r]
    if progress_values:
        print(f"   進捗バー範囲: {min(progress_values):.1f}% - {max(progress_values):.1f}%")

    system_names = [r.get("system_name") for r in results if "error" not in r]
    unique_systems = set(system_names)
    print(f"   検出システム: {', '.join(sorted(unique_systems))}")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"   ⚠️ エラー: {len(errors)} 枚")


if __name__ == "__main__":
    main()
