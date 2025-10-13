"""
ヘッダー画像生成スクリプト
X (Twitter) / note 用のヘッダー画像を自動生成
"""

import os

from PIL import Image, ImageDraw, ImageFont


def create_gradient_background(width, height):
    """青×金のグラデーション背景を作成"""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # 青系のグラデーション
    for y in range(height):
        # 上から下へ: ダークブルー → ミディアムブルー
        r = int(20 + (40 - 20) * y / height)
        g = int(40 + (80 - 40) * y / height)
        b = int(80 + (140 - 80) * y / height)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    return img


def add_circuit_pattern(img, color=(0, 200, 150, 100)):
    """AI回路パターンを追加（緑のアクセント）"""
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    # 回路線（細い緑の線）
    for i in range(5):
        x = width * (0.6 + i * 0.08)
        y_start = height * 0.2
        y_end = height * 0.8
        draw.line([(x, y_start), (x, y_end)], fill=color, width=2)

        # 接続点（小さい円）
        for j in range(3):
            y = height * (0.3 + j * 0.2)
            draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=color)

    # 横線
    for i in range(3):
        y = height * (0.3 + i * 0.2)
        x_start = width * 0.6
        x_end = width * 0.95
        draw.line([(x_start, y), (x_end, y)], fill=color, width=2)

    return img


def add_golden_accents(img):
    """金色のアクセント（小さい円や線）"""
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    # 金色の小さい円（装飾）
    positions = [
        (width * 0.65, height * 0.25),
        (width * 0.75, height * 0.45),
        (width * 0.85, height * 0.65),
        (width * 0.92, height * 0.35),
    ]

    for x, y in positions:
        r = 6
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(255, 200, 50, 150))

    return img


def create_header_image(
    output_path: str,
    icon_path: str,
    width: int = 1500,
    height: int = 500,
    title: str = "AI Narrative Studio",
    icon_height_ratio: float = 0.7,
    mode: str = "standard",
    background: str = "gradient",  # "gradient" | "white"
    draw_patterns: bool = True,
    include_left_icon: bool = True,
) -> str:
    """
    ヘッダー画像を生成

    Args:
        output_path: 出力先パス
        icon_path: アイコン画像のパス
        width: 幅（デフォルト: 1500px for X）
        height: 高さ（デフォルト: 500px for X）
        title: タイトルテキスト
    """
    print(f"🎨 ヘッダー画像生成開始: {width}x{height}px")

    # 1. 背景
    if background == "white":
        img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        print("✅ 白背景作成完了")
    else:
        img = create_gradient_background(width, height).convert("RGBA")
        print("✅ グラデーション背景作成完了")

    # 2. パターン（任意）
    if draw_patterns:
        img = add_circuit_pattern(img)
        print("✅ 回路パターン追加完了")
        img = add_golden_accents(img)
        print("✅ 金色アクセント追加完了")

    # 4. アイコン画像を配置
    if os.path.exists(icon_path):
        icon = Image.open(icon_path).convert("RGBA")

        # アイコンをリサイズ（比率指定）
        icon_height = int(height * icon_height_ratio)
        aspect_ratio = icon.width / icon.height
        icon_width = int(icon_height * aspect_ratio)
        icon = icon.resize((icon_width, icon_height), Image.Resampling.LANCZOS)

        # 配置モード
        if mode == "watermark":
            # 全面に大きく（透過してウォーターマーク風、右側に薄く）
            wm_height = int(height * 1.2)
            wm_width = int(wm_height * aspect_ratio)
            wm = icon.resize((wm_width, wm_height), Image.Resampling.LANCZOS)
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            x_wm = width - wm_width + 80  # 右側に大きくはみ出させる
            y_wm = (height - wm_height) // 2
            overlay.paste(wm, (x_wm, y_wm), wm)
            img = Image.alpha_composite(img.convert("RGBA"), overlay)

            # 左側に小さめのアイコンも置く（ブランド認知）
            if include_left_icon:
                x_pos = 40
                y_pos = (height - icon_height) // 2
                img.paste(icon, (x_pos, y_pos), icon)
        elif mode == "center":
            # 中央配置
            x_pos = (width - icon_width) // 2
            y_pos = (height - icon_height) // 2
            img.paste(icon, (x_pos, y_pos), icon)
        else:
            # 左端に大きめ配置（standard / large_left）
            x_pos = 40
            y_pos = (height - icon_height) // 2
            img.paste(icon, (x_pos, y_pos), icon)

        print(f"✅ アイコン配置完了: {icon_width}x{icon_height}px")
    else:
        print(f"⚠️  アイコンが見つかりません: {icon_path}")

    # 5. タイトルテキスト（空なら描画しない）
    draw = ImageDraw.Draw(img)

    # システムフォント候補（Windows想定）。見つからない場合は自動で次へ。
    font_paths = [
        "C:/Windows/Fonts/segoeui.ttf",  # Segoe UI
        "C:/Windows/Fonts/arial.ttf",  # Arial
        "C:/Windows/Fonts/calibri.ttf",  # Calibri
    ]

    def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        # 最後の手段
        return ImageFont.load_default()

    font = get_font(72)

    if not title:
        img.save(output_path, quality=95)
        print(f"🎉 ヘッダー画像生成完了: {output_path}")
        print(f"📏 サイズ: {width}x{height}px")
        return output_path

    # テキストサイズを自動調整（幅の55%以内に収める）
    max_width = int(width * 0.55)
    base_size = 72
    if mode == "watermark":
        max_width = int(width * 0.5)

    size = base_size
    while True:
        test_font = get_font(size)
        bbox = draw.textbbox((0, 0), title, font=test_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= max_width or size <= 28:
            font = test_font
            text_width, text_height = tw, th
            break
        size -= 4

    # 中央に配置（少し右寄せ）
    offset_x = 100
    if mode == "watermark":
        offset_x = 40
    x = (width - text_width) // 2 + offset_x
    y = (height - text_height) // 2

    # 影を追加（黒、少しずらす）
    shadow_offset = 4
    draw.text(
        (x + shadow_offset, y + shadow_offset),
        title,
        font=font,
        fill=(0, 0, 0, 180),
    )

    # メインテキスト（白×金のグラデーション風）
    draw.text((x, y), title, font=font, fill=(255, 255, 255))

    # 金色のハイライト（少し上にずらして重ねる）
    draw.text((x, y - 2), title, font=font, fill=(255, 220, 100, 100))

    print("✅ タイトルテキスト追加完了")

    # 6. 保存
    img.save(output_path, quality=95)
    print(f"🎉 ヘッダー画像生成完了: {output_path}")
    print(f"📏 サイズ: {width}x{height}px")

    return output_path


def main():
    """メイン処理"""
    # アイコン画像のパス（ダウンロードフォルダから）
    icon_path = r"c:\Users\stair\Downloads\名称未設定のデザイン (2).png"

    # 出力先
    output_dir = r"c:\Users\stair\Downloads"

    # X (Twitter) 用ヘッダー（1500x500）
    twitter_output = os.path.join(output_dir, "header_twitter_1500x500.png")
    create_header_image(
        output_path=twitter_output,
        icon_path=icon_path,
        width=1500,
        height=500,
        title="AI Narrative Studio",
    )

    print("\n" + "=" * 60)

    # note 用ヘッダー（1280x670）
    note_output = os.path.join(output_dir, "header_note_1280x670.png")
    create_header_image(
        output_path=note_output,
        icon_path=icon_path,
        width=1280,
        height=670,
        title="AI Narrative Studio",
    )

    print("\n" + "=" * 60)
    print("✅ すべてのヘッダー画像生成完了！")
    print(f"📁 保存先: {output_dir}")
    print("  - header_twitter_1500x500.png (X用)")
    print("  - header_note_1280x670.png (note用)")


if __name__ == "__main__":
    main()
