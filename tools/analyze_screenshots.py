"""
スクリーンショットから進捗バー・メトリクス情報を抽出するツール

Usage:
    python tools/analyze_screenshots.py screenshots/progress_tracking/progress_*.png
"""

import json
from pathlib import Path
import sys

from PIL import Image


def analyze_screenshot(image_path: Path) -> dict:
    """
    スクリーンショットを解析し、UIの状態を抽出

    Returns:
        dict: {
            "timestamp": "06:52:10",
            "progress_bar_percent": 14,  # 進捗バーの%（目視または色解析）
            "system_name": "System1",
            "metrics": {...}
        }
    """
    try:
        img = Image.open(image_path)
        width, height = img.size

        # 基本情報
        result = {
            "file": image_path.name,
            "image_size": f"{width}x{height}",
            "timestamp": image_path.stem.split("_")[1]
            + "_"
            + image_path.stem.split("_")[2],
        }

        # ここでは画像を開いて基本情報のみ返す
        # OCRやピクセル解析は必要に応じて追加可能
        print(f"✅ {image_path.name}: {width}x{height}px")

        return result

    except Exception as e:
        print(f"❌ {image_path.name}: エラー {e}")
        return {"file": image_path.name, "error": str(e)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/analyze_screenshots.py <image_paths>")
        sys.exit(1)

    results = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.exists():
            result = analyze_screenshot(path)
            results.append(result)
        else:
            print(f"⚠️ ファイルが見つかりません: {arg}")

    # 結果をJSON出力
    output_path = Path("screenshots/progress_tracking/analysis_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📊 解析結果: {output_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
