"""
アイコン加工ユーティリティ
- 背景（白ベース）を透過
- 自動トリミング（余白除去）
- 3サイズ出力: 1024, 512, 400
- 円形マスク版も出力

前提: Pillow がインストール済み
"""

from __future__ import annotations

import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFilter


def floodfill_transparent_bg(img: Image.Image, thresh: int = 10) -> Image.Image:
    """境界の白背景をフラッドフィルで検出し、透明化する。

    - 画像は RGB/RGBA いずれでも可
    - thresh: 背景類似判定の許容値（大きいほど緩い）
    """
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    base = img.convert("RGB").copy()
    w, h = base.size

    mark_color = (1, 2, 3)  # まず使われない色

    # 四隅からフラッドフィル（背景は外周と連結している前提）
    for xy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        ImageDraw.floodfill(base, xy, mark_color, thresh=thresh)

    # 透過マスクを作成（mark_color の領域＝背景）
    mask = Image.new("L", (w, h), 0)
    mask_px = mask.load()
    base_px = base.load()
    if mask_px is None or base_px is None:
        # フォールバック（何もしない）
        return img.convert("RGBA")
    for y in range(h):
        for x in range(w):
            if base_px[x, y] == mark_color:
                mask_px[x, y] = 255

    # 元画像を RGBA にして、背景を透明化
    rgba = img.convert("RGBA")
    rgba_px = rgba.load()
    if rgba_px is None:
        return rgba
    for y in range(h):
        for x in range(w):
            if mask_px[x, y] == 255:
                r, g, b, a = rgba_px[x, y]
                rgba_px[x, y] = (r, g, b, 0)

    return rgba


def autocrop_transparent(img: Image.Image, padding: int = 0) -> Image.Image:
    """透明部分を除去して自動トリミング（padding は残す余白）。"""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bbox = img.getbbox()
    if not bbox:
        return img
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img.width, right + padding)
    bottom = min(img.height, bottom + padding)
    return img.crop((left, top, right, bottom))


def resize_square(img: Image.Image, size: int) -> Image.Image:
    """正方形キャンバスに収める（余白は透明）。"""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    ratio = min(size / img.width, size / img.height)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(scaled, ((size - new_w) // 2, (size - new_h) // 2), scaled)
    return canvas


def circle_mask(
    img: Image.Image,
    stroke_px: int = 0,
    stroke_color: Tuple[int, int, int, int] = (0, 0, 0, 180),
) -> Image.Image:
    """円形マスクを適用。任意で外枠ストロークを追加。"""
    size = min(img.width, img.height)
    img_sq = resize_square(img, size)

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size - 1, size - 1), fill=255)

    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img_sq, (0, 0), mask)

    if stroke_px > 0:
        # 太らせて外枠を作る
        edge = mask.filter(ImageFilter.MaxFilter(stroke_px * 2 + 1))
        border = Image.new("RGBA", (size, size), stroke_color)
        # 境界のみ残す
        inner = mask
        edge_only = Image.new("L", (size, size), 0)
        e_px, i_px, eo_px = edge.load(), inner.load(), edge_only.load()
        if e_px is None or i_px is None or eo_px is None:
            return out
        for y in range(size):
            for x in range(size):
                val = e_px[x, y] - i_px[x, y]
                if val < 0:
                    val = 0
                eo_px[x, y] = val
        out = Image.composite(border, out, edge_only)
    return out


def process_icon(
    src_path: str,
    out_dir: str,
    sizes: Tuple[int, ...] = (1024, 512, 400),
    circle: bool = True,
    stroke_px: int = 3,
) -> None:
    print(f"📥 入力: {src_path}")
    img = Image.open(src_path)

    # 1) 背景透過
    img = floodfill_transparent_bg(img, thresh=15)

    # 2) 自動トリミング
    img = autocrop_transparent(img, padding=8)

    os.makedirs(out_dir, exist_ok=True)

    for s in sizes:
        sq = resize_square(img, s)
        out_path = os.path.join(out_dir, f"icon_clean_{s}.png")
        sq.save(out_path)
        print(f"✅ 保存: {out_path}")

        if circle:
            circ = circle_mask(sq, stroke_px=stroke_px)
            out_path_c = os.path.join(out_dir, f"icon_circle_{s}.png")
            circ.save(out_path_c)
            print(f"✅ 保存: {out_path_c}")

    print("🎉 すべて完了！")


if __name__ == "__main__":
    # 既定の入力アイコン（ダウンロードフォルダ）
    SRC = r"c:\Users\stair\Downloads\名称未設定のデザイン (2).png"
    OUT = r"c:\Users\stair\Downloads\processed_icon"
    process_icon(SRC, OUT)
