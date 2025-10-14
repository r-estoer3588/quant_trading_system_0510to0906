"""
note記事用アイキャッチ画像生成スクリプト
対話形式の記事に最適化したデザイン

機能:
- デフォルト: シルエット（円＋吹き出し）でユイ/レンを描画
- オプション: 透過PNGの人物イラストを差し替え配置 (--yui, --ren)
- タイトル・サイズ・出力先の上書き指定 (--title1, --title2, --width, --height, --out)
"""

import argparse
import os
import sys

from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    ImageOps,
)


def create_gradient_background(width, height):
    """ブランドカラーのグラデーション背景"""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # ダークブルー (#142A4D) → ブルー (#2C507A)
    for y in range(height):
        ratio = y / height
        r = int(20 + (44 - 20) * ratio)
        g = int(42 + (80 - 42) * ratio)
        b = int(77 + (122 - 77) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    return img


def add_circuit_pattern(img, color=(0, 200, 150, 80)):
    """AI回路パターン（控えめ）"""
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    # 右側に縦線
    for i in range(4):
        x = int(width * (0.7 + i * 0.07))
        y_start = int(height * 0.2)
        y_end = int(height * 0.8)
        draw.line([(x, y_start), (x, y_end)], fill=color, width=2)

        # 接続点
        for j in range(3):
            y = int(height * (0.35 + j * 0.15))
            draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=color)

    # 横線
    for i in range(3):
        y = int(height * (0.35 + i * 0.15))
        x_start = int(width * 0.7)
        x_end = int(width * 0.95)
        draw.line([(x_start, y), (x_end, y)], fill=color, width=2)

    return img


def draw_dialogue_characters(img):
    """対話キャラクターのシルエット（左右に配置）"""
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    # ユイ（左側、初心者エンジニア）- 小さめの円
    yui_x = int(width * 0.12)
    yui_y = int(height * 0.5)
    yui_radius = int(height * 0.15)

    # 頭部（円）
    draw.ellipse(
        [
            (yui_x - yui_radius, yui_y - yui_radius),
            (yui_x + yui_radius, yui_y + yui_radius),
        ],
        fill=(255, 200, 100, 200),  # 金色系
        outline=(255, 220, 120, 255),
        width=3,
    )

    # 吹き出し（右上）
    bubble_x = yui_x + int(yui_radius * 1.8)
    bubble_y = yui_y - int(yui_radius * 1.2)
    bubble_w = int(width * 0.08)
    bubble_h = int(height * 0.08)

    draw.ellipse(
        [
            (bubble_x, bubble_y),
            (bubble_x + bubble_w, bubble_y + bubble_h),
        ],
        fill=(255, 255, 255, 220),
        outline=(255, 200, 100, 255),
        width=2,
    )

    # 吹き出しの尾（小さい円2つ）
    tail1_x = bubble_x - int(bubble_w * 0.15)
    tail1_y = bubble_y + int(bubble_h * 0.6)
    draw.ellipse(
        [(tail1_x, tail1_y), (tail1_x + 8, tail1_y + 8)],
        fill=(255, 255, 255, 200),
    )

    tail2_x = tail1_x - 10
    tail2_y = tail1_y + 8
    draw.ellipse(
        [(tail2_x, tail2_y), (tail2_x + 5, tail2_y + 5)],
        fill=(255, 255, 255, 180),
    )

    # 吹き出し内のテキスト（「？」）
    font = get_font(int(bubble_h * 0.6))
    question_text = "?"
    bbox = draw.textbbox((0, 0), question_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = bubble_x + (bubble_w - text_w) // 2
    text_y = bubble_y + (bubble_h - text_h) // 2
    draw.text((text_x, text_y), question_text, font=font, fill=(100, 100, 100, 255))

    # レン先輩（右側、ベテラン）- 大きめの円
    ren_x = int(width * 0.88)
    ren_y = int(height * 0.5)
    ren_radius = int(height * 0.18)

    # 頭部（円）
    draw.ellipse(
        [
            (ren_x - ren_radius, ren_y - ren_radius),
            (ren_x + ren_radius, ren_y + ren_radius),
        ],
        fill=(0, 200, 150, 200),  # 緑系
        outline=(0, 220, 170, 255),
        width=3,
    )

    # 吹き出し（左上）
    bubble2_x = ren_x - int(ren_radius * 2.5)
    bubble2_y = ren_y - int(ren_radius * 1.3)
    bubble2_w = int(width * 0.1)
    bubble2_h = int(height * 0.08)

    draw.ellipse(
        [
            (bubble2_x, bubble2_y),
            (bubble2_x + bubble2_w, bubble2_y + bubble2_h),
        ],
        fill=(255, 255, 255, 220),
        outline=(0, 200, 150, 255),
        width=2,
    )

    # 吹き出しの尾
    tail3_x = bubble2_x + bubble2_w
    tail3_y = bubble2_y + int(bubble2_h * 0.7)
    draw.ellipse(
        [(tail3_x, tail3_y), (tail3_x + 8, tail3_y + 8)],
        fill=(255, 255, 255, 200),
    )

    tail4_x = tail3_x + 10
    tail4_y = tail3_y + 8
    draw.ellipse(
        [(tail4_x, tail4_y), (tail4_x + 5, tail4_y + 5)],
        fill=(255, 255, 255, 180),
    )

    # 吹き出し内のテキスト（「!」）
    exclaim_text = "!"
    bbox2 = draw.textbbox((0, 0), exclaim_text, font=font)
    text2_w = bbox2[2] - bbox2[0]
    text2_h = bbox2[3] - bbox2[1]
    text2_x = bubble2_x + (bubble2_w - text2_w) // 2
    text2_y = bubble2_y + (bubble2_h - text2_h) // 2
    draw.text((text2_x, text2_y), exclaim_text, font=font, fill=(100, 100, 100, 255))

    return img


def paste_character(
    base: Image.Image,
    char_img_path: str,
    center_xy: tuple[int, int],
    max_height_ratio: float = 0.5,
    add_shadow: bool = True,
    unify_style: bool = False,
    posterize_bits: int = 4,
    saturation: float = 0.95,
    contrast: float = 1.05,
    tint_rgba: tuple[int, int, int, int] | None = None,
    outline_px: int = 0,
    outline_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> Image.Image:
    """人物イラストPNGを読み込み、中心座標に最大高さ比で貼り付ける。

    Args:
        base: 背景画像 (RGBA)
        char_img_path: 透過PNGのパス
        center_xy: (x, y) の中心座標（貼り付け位置）
        max_height_ratio: 画像高さに対する最大高さ比（0-1）
        add_shadow: 影を付与するか
    """
    if not os.path.exists(char_img_path):
        return base

    bg = base.convert("RGBA")
    width, height = bg.size

    try:
        char_img = Image.open(char_img_path).convert("RGBA")
    except Exception:
        return base

    # リサイズ（高さを max_height_ratio に合わせる）
    target_h = int(height * max_height_ratio)
    aspect = char_img.width / char_img.height if char_img.height else 1.0
    target_w = max(1, int(target_h * aspect))
    char_resized = char_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    def apply_unify_style(img_rgba: Image.Image) -> Image.Image:
        # RGBA分離
        rgb = img_rgba.convert("RGB")
        alpha = img_rgba.split()[-1]
        # 1) ポスタライズ（階調削減でフラット感）
        try:
            rgb = ImageOps.posterize(rgb, posterize_bits)
        except Exception:
            pass
        # 2) 彩度・コントラスト微調整
        try:
            rgb = ImageEnhance.Color(rgb).enhance(saturation)
            rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
        except Exception:
            pass
        # 3) ティント（薄く色を載せる）
        if tint_rgba is not None:
            overlay = Image.new("RGBA", rgb.size, tint_rgba)
            # 低アルファでブレンド（上から重ねる）
            rgb = Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")
        # RGBAへ戻す
        out = rgb.convert("RGBA")
        out.putalpha(alpha)
        return out

    def add_outline(
        img_rgba: Image.Image,
        px: int,
        color: tuple[int, int, int, int],
    ) -> Image.Image:
        if px <= 0:
            return img_rgba
        alpha = img_rgba.split()[-1]
        # 膨張（MaxFilter）で外側に広げる
        size = max(3, px * 2 + 1)
        expanded = alpha.filter(ImageFilter.MaxFilter(size=size))
        # 輪郭 = expanded - original
        try:
            outline_mask = ImageChops.subtract(expanded, alpha)
        except Exception:
            # 代替: expanded をそのまま利用
            outline_mask = expanded
        outline_img = Image.new("RGBA", img_rgba.size, color)
        outline_img.putalpha(outline_mask)
        base_layer = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
        base_layer = Image.alpha_composite(base_layer, outline_img)
        base_layer = Image.alpha_composite(base_layer, img_rgba)
        return base_layer

    # スタイル統一処理
    if unify_style:
        char_resized = apply_unify_style(char_resized)
        if outline_px > 0:
            char_resized = add_outline(char_resized, outline_px, outline_color)

    # 影（アルファマスクを使ってドロップシャドウ風）
    if add_shadow:
        # 影用に黒塗りマスク作成
        shadow = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        shadow.paste(char_resized.split()[-1], (0, 0))  # alpha を流用
        shadow = shadow.filter(ImageFilter.GaussianBlur(6))
        shadow_colored = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 120))
        shadow_colored.putalpha(shadow.split()[-1])

    cx, cy = center_xy
    paste_x = int(cx - target_w / 2)
    paste_y = int(cy - target_h / 2)

    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))

    if add_shadow:
        overlay.paste(shadow_colored, (paste_x + 6, paste_y + 6), shadow_colored)

    overlay.paste(char_resized, (paste_x, paste_y), char_resized)
    composed = Image.alpha_composite(bg, overlay)
    return composed


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """フォント取得（Windows用・おしゃれフォント優先）"""
    font_paths = [
        "C:/Windows/Fonts/YuGothB.ttc",  # Yu Gothic UI Bold（游ゴシック太字・モダン）
        "C:/Windows/Fonts/YuGothM.ttc",  # Yu Gothic UI Medium
        "C:/Windows/Fonts/yugothib.ttf",  # Yu Gothic Bold
        "C:/Windows/Fonts/meiryob.ttc",  # Meiryo Bold（日本語太字）
        "C:/Windows/Fonts/segoeuib.ttf",  # Segoe UI Bold
        "C:/Windows/Fonts/msgothic.ttc",  # MS Gothic（日本語）
        "C:/Windows/Fonts/meiryo.ttc",  # Meiryo（日本語）
        "C:/Windows/Fonts/arialbd.ttf",  # Arial Bold
    ]

    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue

    return ImageFont.load_default()


def add_title_text(img, title_lines):
    """タイトルテキストを追加（上部配置、半透明背景付き）"""
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # フォントサイズ（2行想定）
    font_large = get_font(80)
    font_medium = get_font(60)

    # 1行目のサイズ計算
    line1 = title_lines[0]
    bbox1 = draw.textbbox((0, 0), line1, font=font_large)
    text1_w = bbox1[2] - bbox1[0]
    text1_h = bbox1[3] - bbox1[1]

    # 2行目のサイズ計算
    line2 = title_lines[1]
    bbox2 = draw.textbbox((0, 0), line2, font=font_medium)
    text2_w = bbox2[2] - bbox2[0]
    text2_h = bbox2[3] - bbox2[1]

    # 上部5%の位置に配置（帯を完全に避ける）
    y1 = int(height * 0.05)
    y2 = y1 + text1_h + 20

    # 半透明の黒背景ボックスを追加（可読性向上）
    bg_padding = 30
    bg_box = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg_box)
    bg_y_start = y1 - bg_padding
    bg_y_end = y2 + text2_h + bg_padding
    bg_draw.rectangle(
        [(0, bg_y_start), (width, bg_y_end)], fill=(0, 0, 0, 100)  # 黒、透明度100/255
    )
    img = Image.alpha_composite(img.convert("RGBA"), bg_box)

    # 新しいdrawオブジェクトを作成
    draw = ImageDraw.Draw(img)

    # 1行目を中央配置
    x1 = (width - text1_w) // 2

    # 影
    draw.text((x1 + 3, y1 + 3), line1, font=font_large, fill=(0, 0, 0, 180))
    # メインテキスト
    draw.text((x1, y1), line1, font=font_large, fill=(255, 255, 255, 255))

    # 2行目を中央配置
    x2 = (width - text2_w) // 2

    # 影
    draw.text((x2 + 3, y2 + 3), line2, font=font_medium, fill=(0, 0, 0, 180))
    # メインテキスト（金色のアクセント）
    draw.text((x2, y2), line2, font=font_medium, fill=(255, 220, 100, 255))

    return img


def create_article_eyecatch(
    output_path: str,
    title_line1: str = "対話で学ぶ",
    title_line2: str = "Playwright × AI",
    width: int = 1280,
    height: int = 670,
    yui_path: str | None = None,
    ren_path: str | None = None,
    disable_circuit: bool = False,
    unify_style: bool = False,
    posterize_bits: int = 4,
    saturation: float = 0.95,
    contrast: float = 1.05,
    tint: str | None = None,
    outline_px: int = 0,
    add_shadow: bool = True,
) -> str:
    """
    note記事用アイキャッチ画像を生成

    Args:
        output_path: 出力先パス
        title_line1: タイトル1行目
        title_line2: タイトル2行目
        width: 幅（デフォルト: 1280px、note推奨）
        height: 高さ（デフォルト: 670px、note推奨）

    Returns:
        生成された画像のパス
    """
    print(f"🎨 アイキャッチ画像生成開始: {width}x{height}px")

    # 1. グラデーション背景
    img = create_gradient_background(width, height).convert("RGBA")
    print("✅ グラデーション背景作成完了")

    # 2. AI回路パターン（右側）
    if not disable_circuit:
        img = add_circuit_pattern(img)
        print("✅ 回路パターン追加完了")

    # 3. 対話キャラクター
    if yui_path or ren_path:
        # 人物イラストが指定された場合はそれを優先
        w, h = img.size
        # ティント色（ブランド系）
        tint_map = {
            "navy": (44, 80, 122, 40),  # #2C507A, alpha 40
            "teal": (0, 200, 150, 32),  # #00C896, alpha 32
            "gold": (255, 200, 50, 24),
        }
        tint_rgba = tint_map.get((tint or "").lower()) if tint else None
        # 左: ユイ（高さ50%） 位置は左1/6付近
        if yui_path:
            img = paste_character(
                img,
                yui_path,
                center_xy=(int(w * 0.14), int(h * 0.58)),
                max_height_ratio=0.5,
                add_shadow=add_shadow,
                unify_style=unify_style,
                posterize_bits=posterize_bits,
                saturation=saturation,
                contrast=contrast,
                tint_rgba=tint_rgba,
                outline_px=outline_px,
            )
        # 右: レン先輩（高さ58%） 位置は右5/6付近
        if ren_path:
            img = paste_character(
                img,
                ren_path,
                center_xy=(int(w * 0.86), int(h * 0.56)),
                max_height_ratio=0.58,
                add_shadow=add_shadow,
                unify_style=unify_style,
                posterize_bits=posterize_bits,
                saturation=saturation,
                contrast=contrast,
                tint_rgba=tint_rgba,
                outline_px=outline_px,
            )
        print("✅ 人物イラスト配置完了")
    else:
        # フォールバック: シルエット描画
        img = draw_dialogue_characters(img)
        print("✅ キャラクターシルエット追加完了")

    # 4. タイトルテキスト（中央）
    if title_line1 or title_line2:  # タイトルが指定されている場合のみ描画
        img = add_title_text(img, [title_line1, title_line2])
        print("✅ タイトルテキスト追加完了")
    else:
        print("ℹ️  タイトルなし版を生成（ベース画像）")

    # 5. 保存
    img = img.convert("RGB")  # RGBA → RGB（JPEG互換）
    img.save(output_path, quality=95)
    print(f"🎉 アイキャッチ画像生成完了: {output_path}")
    print(f"📏 サイズ: {width}x{height}px")

    return output_path


def main():
    """メイン処理（CLI対応）"""
    parser = argparse.ArgumentParser(description="note記事用アイキャッチ画像生成")
    parser.add_argument(
        "--out", dest="out", default=None, help="出力ファイルパス (png)"
    )
    parser.add_argument(
        "--title1", dest="title1", default="対話で学ぶ", help="タイトル1行目"
    )
    parser.add_argument(
        "--title2",
        dest="title2",
        default="Playwright × AI",
        help="タイトル2行目",
    )
    parser.add_argument(
        "--no-title",
        dest="no_title",
        action="store_true",
        help="タイトルを描画しない（ベース画像生成用）",
    )
    parser.add_argument(
        "--base",
        dest="base_image",
        default=None,
        help="ベース画像のパス（既存画像にタイトルのみ追加）",
    )
    parser.add_argument("--width", dest="width", type=int, default=1280, help="画像幅")
    parser.add_argument(
        "--height", dest="height", type=int, default=670, help="画像高さ"
    )
    parser.add_argument(
        "--yui", dest="yui", default=None, help="ユイ画像のパス (透過PNG)"
    )
    parser.add_argument(
        "--ren", dest="ren", default=None, help="レン先輩画像のパス (透過PNG)"
    )
    parser.add_argument(
        "--no-circuit",
        dest="no_circuit",
        action="store_true",
        help="回路パターンを描画しない",
    )
    # スタイル統一系
    parser.add_argument(
        "--unify-style",
        dest="unify_style",
        action="store_true",
        help="人物画像にフラット化/色味統一を適用",
    )
    parser.add_argument(
        "--posterize-bits",
        dest="posterize_bits",
        type=int,
        default=4,
        help="階調削減ビット数 (小さいほどフラット) 例:4",
    )
    parser.add_argument(
        "--saturation",
        dest="saturation",
        type=float,
        default=0.95,
        help="彩度係数 (1.0で変化なし)",
    )
    parser.add_argument(
        "--contrast",
        dest="contrast",
        type=float,
        default=1.05,
        help="コントラスト係数 (1.0で変化なし)",
    )
    parser.add_argument(
        "--tint",
        dest="tint",
        default=None,
        choices=["navy", "teal", "gold"],
        help="薄い色味を重ねて統一 (navy/teal/gold)",
    )
    parser.add_argument(
        "--stroke",
        dest="outline_px",
        type=int,
        default=0,
        help="人物の外周に白フチ(px)を追加してタッチ差を緩和",
    )
    parser.add_argument(
        "--no-shadow",
        dest="no_shadow",
        action="store_true",
        help="人物ドロップシャドウを無効化",
    )

    args = parser.parse_args()

    # タイトルなしオプションの処理
    title1 = "" if args.no_title else args.title1
    title2 = "" if args.no_title else args.title2

    # 出力先
    output_path = args.out or os.path.join(
        os.path.expanduser("~"),
        "Downloads",
        "eyecatch_playwright_ai.png",
    )

    # ベース画像モード: 既存画像にタイトルのみ追加
    if args.base_image:
        if not os.path.exists(args.base_image):
            print(f"❌ エラー: ベース画像が見つかりません: {args.base_image}")
            sys.exit(1)

        print(f"📂 ベース画像を読み込み: {args.base_image}")
        base_img = Image.open(args.base_image).convert("RGBA")

        # Tint適用（指定された場合）
        if args.tint and args.tint != "none":
            print(f"🎨 Tint適用: {args.tint}")
            tint_map = {
                "navy": (44, 80, 122, 40),
                "teal": (0, 200, 150, 32),
                "gold": (255, 200, 50, 24),
            }
            tint_rgba = tint_map.get(args.tint.lower())
            if tint_rgba:
                overlay = Image.new("RGBA", base_img.size, tint_rgba)
                base_img = Image.alpha_composite(base_img, overlay)
                print(f"✅ Tint ({args.tint}) 適用完了")

        # タイトル追加
        if title1 or title2:
            final_img = add_title_text(base_img, [title1, title2])
            print("✅ タイトルテキスト追加完了")
        else:
            final_img = base_img
            print("ℹ️ タイトルなし（ベース画像のみ）")

        # 保存
        final_img.convert("RGB").save(output_path, "PNG")
        print(f"✅ 画像保存完了: {output_path}")

    else:
        # 通常モード: キャラクター配置から生成
        create_article_eyecatch(
            output_path=output_path,
            title_line1=title1,
            title_line2=title2,
            width=args.width,
            height=args.height,
            yui_path=args.yui,
            ren_path=args.ren,
            disable_circuit=args.no_circuit,
            unify_style=args.unify_style,
            posterize_bits=args.posterize_bits,
            saturation=args.saturation,
            contrast=args.contrast,
            tint=args.tint,
            outline_px=args.outline_px,
        )

    print("\n" + "=" * 60)
    print("✅ アイキャッチ画像生成完了！")
    print(f"📁 保存先: {output_path}")
    print("\n📝 note記事への挿入方法:")
    print("  1. note編集画面を開く")
    print("  2. 記事の最初（タイトル直後）に画像を挿入")
    print("  3. この画像をアップロード")
    print("  4. 保存すると自動的にアイキャッチとして使われます")


if __name__ == "__main__":
    main()
