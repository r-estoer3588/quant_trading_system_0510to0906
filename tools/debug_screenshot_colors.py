"""
スクリーンショットの色分布を調査するデバッグツール
"""

import sys
from pathlib import Path

from PIL import Image


def analyze_color_distribution(image_path: Path):
    """画像の色分布を分析"""
    img = Image.open(image_path)
    width, height = img.size

    # 上部15%領域
    top_h = int(height * 0.15)
    top_region = img.crop((0, 0, width, top_h))

    pixels = list(top_region.getdata())

    # RGB値の分布を集計
    color_counts = {}
    for pixel in pixels:
        if len(pixel) >= 3:
            r, g, b = pixel[:3]
            # 10刻みで丸めて集計
            r_bucket = (r // 10) * 10
            g_bucket = (g // 10) * 10
            b_bucket = (b // 10) * 10
            key = (r_bucket, g_bucket, b_bucket)
            color_counts[key] = color_counts.get(key, 0) + 1

    # 上位10色を表示
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n📸 {image_path.name}")
    print(f"   サイズ: {width}x{height}px")
    print(f"   上部領域: {width}x{top_h}px ({len(pixels)} ピクセル)\n")
    print("   上位10色（RGB, ピクセル数）:")

    for i, ((r, g, b), count) in enumerate(sorted_colors[:10], 1):
        percent = (count / len(pixels)) * 100
        print(
            f"   {i:2d}. RGB({r:3d}, {g:3d}, {b:3d}): {count:6d} px ({percent:5.1f}%)"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/debug_screenshot_colors.py <image_path>")
        sys.exit(1)

    for arg in sys.argv[1:]:
        analyze_color_distribution(Path(arg))
