"""
スクリーンショット簡易解析ツール（Pillow のみ使用）

進捗バーのピクセル解析で%を推定。OCRなしでも基本分析が可能。

Usage:
    python tools/analyze_screenshots_simple.py screenshots/progress_tracking/*.png
"""

import json
from pathlib import Path
import re
import sys

from PIL import Image


def extract_timestamp_from_filename(filename: str) -> str:
    """ファイル名からタイムスタンプを抽出"""
    match = re.match(r"progress_(\d{8})_(\d{6})_\d+\.png", filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        return f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
    return ""


def analyze_progress_bar_pixels(img: Image.Image) -> dict:
    """
    進捗バーをピクセル解析で推定

    Streamlitの進捗バーは画面上部に表示され、
    特定の色（通常は青系 rgb(29, 233, 182) や緑系）で塗りつぶされる。
    """
    try:
        width, height = img.size

        # 上部15%領域を解析対象
        top_h = int(height * 0.15)
        top_region = img.crop((0, 0, width, top_h))

        # RGB統計を取得
        pixels = list(top_region.getdata())

        # Streamlit進捗バーの色を検出
        # 青緑系: rgb(29, 233, 182) 近辺
        # オレンジ系（警告）: rgb(255, 75, 75)
        progress_pixels = 0
        total_pixels = len(pixels)

        for pixel in pixels:
            if len(pixel) >= 3:
                r, g, b = pixel[:3]
                # 青緑系の進捗バー色
                if 20 < r < 100 and 200 < g < 255 and 150 < b < 220:
                    progress_pixels += 1
                # 緑系の進捗バー色
                elif g > 180 and g > r + 50 and g > b + 30:
                    progress_pixels += 1

        # 進捗バーの占有率から%を推定
        # 画面幅の約90%が進捗バー領域と仮定
        bar_width_ratio = 0.9
        estimated_progress = (progress_pixels / (total_pixels * bar_width_ratio)) * 100

        return {
            "progress_percent": min(100.0, max(0.0, estimated_progress)),
            "progress_pixels": progress_pixels,
            "total_pixels": total_pixels,
        }
    except Exception as e:
        return {"progress_percent": 0.0, "error": str(e)}


def analyze_screenshot(image_path: Path) -> dict:
    """スクリーンショット1枚を解析"""
    try:
        img = Image.open(image_path)
        width, height = img.size

        timestamp = extract_timestamp_from_filename(image_path.name)
        progress_data = analyze_progress_bar_pixels(img)

        result = {
            "file": image_path.name,
            "timestamp": timestamp,
            "progress_percent": round(progress_data["progress_percent"], 1),
            "image_size": f"{width}x{height}",
            "raw_pixel_count": progress_data.get("progress_pixels", 0),
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
        print("Usage: python tools/analyze_screenshots_simple.py <image_paths>")
        sys.exit(1)

    print("🔍 スクリーンショット簡易解析開始...")
    print(f"   対象: {len(sys.argv) - 1} 枚\n")

    results = []
    for i, arg in enumerate(sys.argv[1:], 1):
        path = Path(arg)
        if path.exists() and path.suffix.lower() == ".png":
            result = analyze_screenshot(path)
            results.append(result)

            if i % 50 == 0 or i == len(sys.argv) - 1:
                print(f"   処理中: {i}/{len(sys.argv) - 1} 枚...")

    # 結果をJSON出力
    output_path = Path("screenshots/progress_tracking/analysis_simple.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 解析完了: {len(results)} 枚")
    print(f"📊 結果: {output_path}\n")

    # サマリー表示
    print("【サマリー】")
    valid_results = [r for r in results if "error" not in r]
    progress_values = [r["progress_percent"] for r in valid_results]

    if progress_values:
        print(
            f"   進捗バー範囲: {min(progress_values):.1f}% - {max(progress_values):.1f}%"
        )
        print(f"   平均進捗: {sum(progress_values) / len(progress_values):.1f}%")

        # タイムスタンプ範囲
        timestamps = [r["timestamp"] for r in valid_results if r["timestamp"]]
        if timestamps:
            print(f"   時刻範囲: {min(timestamps)} - {max(timestamps)}")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"   ⚠️ エラー: {len(errors)} 枚")


if __name__ == "__main__":
    main()
