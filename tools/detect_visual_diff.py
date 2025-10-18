"""
Playwright による連続スクショ間のビジュアル差分検出ツール

進捗バー後退、UI要素の位置変化、サイズ変化などを検出。

Usage:
    python tools/detect_visual_diff.py \
        --input screenshots/progress_tracking \
        --output visual_diff_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops


def calculate_image_diff(img1: Image.Image, img2: Image.Image) -> dict[str, Any]:
    """
    2枚の画像の差分を計算

    Returns:
        dict: {
            "diff_percentage": float,  # 差分ピクセル比率 (%)
            "changed_pixels": int,
            "total_pixels": int,
        }
    """
    # 画像サイズが異なる場合はリサイズ
    if img1.size != img2.size:
        # 小さい方に合わせる
        target_size = (
            min(img1.width, img2.width),
            min(img1.height, img2.height),
        )
        img1 = img1.resize(target_size)
        img2 = img2.resize(target_size)

    # 差分画像を生成
    diff = ImageChops.difference(img1, img2)

    # 差分ピクセルをカウント
    changed_pixels = 0
    total_pixels = diff.width * diff.height

    for x in range(diff.width):
        for y in range(diff.height):
            pixel = diff.getpixel((x, y))
            if isinstance(pixel, tuple):
                # RGBの差分が閾値以上なら変化したとみなす
                if max(pixel[:3]) > 30:  # 閾値: 30/255
                    changed_pixels += 1
            elif pixel > 30:
                changed_pixels += 1

    diff_percentage = (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    return {
        "diff_percentage": round(diff_percentage, 2),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
    }


def detect_progress_bar_region_change(img1: Image.Image, img2: Image.Image) -> dict[str, Any]:
    """
    進捗バー領域の変化を検出（色ヒストグラム比較）

    進捗バーが後退した場合、緑色領域が減少する。
    """
    width, height = img1.size

    # 進捗バー推定領域（画像上部）
    progress_region = (
        int(width * 0.1),
        int(height * 0.05),
        int(width * 0.9),
        int(height * 0.15),
    )

    region1 = img1.crop(progress_region)
    region2 = img2.crop(progress_region)

    # 緑色ピクセル数をカウント
    def count_green_pixels(img: Image.Image) -> int:
        count = 0
        for x in range(img.width):
            for y in range(img.height):
                pixel = img.getpixel((x, y))
                if isinstance(pixel, tuple) and len(pixel) >= 3:
                    r, g, b = pixel[:3]
                    # Streamlit進捗バーの緑色
                    if g > 150 and r < 100 and b < 100:
                        count += 1
        return count

    green1 = count_green_pixels(region1)
    green2 = count_green_pixels(region2)

    # 進捗後退判定
    regression = green2 < green1 * 0.8  # 20%以上減少

    return {
        "green_pixels_prev": green1,
        "green_pixels_curr": green2,
        "progress_regression_detected": regression,
        "green_pixel_change": green2 - green1,
    }


def analyze_screenshot_pair(prev_path: Path, curr_path: Path) -> dict[str, Any]:
    """連続する2枚のスクショを比較"""
    try:
        img1 = Image.open(prev_path)
        img2 = Image.open(curr_path)

        # 全体差分
        overall_diff = calculate_image_diff(img1, img2)

        # 進捗バー領域の変化
        progress_bar_diff = detect_progress_bar_region_change(img1, img2)

        result = {
            "prev_file": prev_path.name,
            "curr_file": curr_path.name,
            "overall_diff": overall_diff,
            "progress_bar_diff": progress_bar_diff,
            "status": "ok",
        }

        # 進捗後退検出
        if progress_bar_diff["progress_regression_detected"]:
            result["status"] = "progress_regression"
            print(
                f"🔴 進捗後退検出: {prev_path.name} → {curr_path.name} "
                f"(緑ピクセル: {progress_bar_diff['green_pixels_prev']} → "
                f"{progress_bar_diff['green_pixels_curr']})"
            )

        # 大きな変化検出
        elif overall_diff["diff_percentage"] > 10:
            result["status"] = "major_change"
            print(f"⚠️  大きな変化: {prev_path.name} → {curr_path.name} ({overall_diff['diff_percentage']}%)")

        return result

    except Exception as e:
        print(f"❌ 比較エラー: {prev_path.name} vs {curr_path.name}: {e}")
        return {
            "prev_file": prev_path.name,
            "curr_file": curr_path.name,
            "error": str(e),
            "status": "error",
        }


def main():
    parser = argparse.ArgumentParser(description="ビジュアル差分検出")
    parser.add_argument(
        "--input",
        type=str,
        default="screenshots/progress_tracking",
        help="スクショディレクトリ",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="screenshots/progress_tracking/visual_diff_report.json",
        help="差分レポート出力先",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="比較ペア数上限（テスト用）",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_file = Path(args.output)

    # スクショファイル一覧取得
    screenshot_files = sorted(input_dir.glob("progress_*.png"))

    if args.limit:
        screenshot_files = screenshot_files[: args.limit + 1]

    print(f"🔍 ビジュアル差分検出開始: {len(screenshot_files)} 枚")
    print(f"📂 入力: {input_dir}")
    print(f"📄 出力: {output_file}")
    print("")

    # 連続ペア比較
    diff_results = []
    for i in range(1, len(screenshot_files)):
        prev_path = screenshot_files[i - 1]
        curr_path = screenshot_files[i]

        result = analyze_screenshot_pair(prev_path, curr_path)
        diff_results.append(result)

    # サマリー
    progress_regressions = [r for r in diff_results if r.get("status") == "progress_regression"]
    major_changes = [r for r in diff_results if r.get("status") == "major_change"]

    report = {
        "summary": {
            "total_pairs": len(diff_results),
            "progress_regressions": len(progress_regressions),
            "major_changes": len(major_changes),
        },
        "diff_results": diff_results,
        "regression_details": progress_regressions,
    }

    # JSON出力
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("")
    print(f"✅ 差分検出完了: {len(diff_results)} ペア")
    print(f"📊 結果: {output_file}")
    print(f"🔴 進捗後退: {len(progress_regressions)} 件")
    print(f"⚠️  大きな変化: {len(major_changes)} 件")


if __name__ == "__main__":
    main()
