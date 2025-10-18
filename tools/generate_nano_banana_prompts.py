"""
Nano Banana 用の一行プロンプトを生成する小さなユーティリティ

使い方:
  python tools/generate_nano_banana_prompts.py --title "対話で学ぶ Playwright の CI 失敗を 30 分で潰す実践ガイド"

出力: 標準出力に 4-5 行の短いプロンプトとネガティブプロンプトを出力する。
オプション --out を指定すると `docs/nano_banana_prompts_{slug}.txt` に保存する。

方針: Note のアイキャッチ向けに、記事タイトルを元に複数のスタイル (コンソール/キャラクター/抽象/タイポグラフィ) を生成します。
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Iterable, List


def slugify(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"^-+|-+$", "", s)
    return s or "prompt"


def build_prompts(title: str) -> List[str]:
    """タイトルから複数スタイルの Nano Banana 一行プロンプトを作る"""
    title_short = title.replace("#", "").strip()

    base_meta = "1280x670, high detail, clean composition, flat colors"

    prompts = [
        # A: コンソール / ログ分析にフォーカス
        (
            f"{title_short} — a focused developer scene showing a laptop screen "
            f"with console logs and highlighted error lines, shallow depth of "
            f"field, cool blue gradient background, subtle speech bubble, "
            f"{base_meta}"
        ),
        # B: キャラクター対話風（フラットイラスト）
        (
            f"{title_short} — two flat-style characters (junior and senior) "
            f"in conversation, speech bubbles with simple icons, friendly "
            f"illustration, warm accent color, {base_meta}"
        ),
        # C: 抽象的な可視化（ログ解析 / グラフ）
        (
            f"{title_short} — abstract visualization of logs and metrics, "
            f"stylized waveform and highlighted failure point, modern "
            f"infographic style, teal and navy palette, {base_meta}"
        ),
        # D: タイポグラフィ中心（タイトルを大胆に扱う）
        (
            f"{title_short} — bold typographic composition of the title, "
            f"large readable Japanese text, minimal background, subtle "
            f"texture, high contrast, {base_meta}"
        ),
        # E: シネマティック写真風（現場感）
        (
            f"{title_short} — cinematic photo-style of an engineer at a "
            f"desk with a laptop showing test failures, moody lighting, "
            f"cinematic crop, {base_meta}"
        ),
    ]

    negative = "lowres, watermark, signature, extra text, deformed, blurry, " "oversaturated"

    # Ensure short lines: strip repeated whitespace
    prompts = [re.sub(r"\s+", " ", p).strip() for p in prompts]
    return prompts + [f"Negative prompt: {negative}"]


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=("Generate Nano Banana single-line prompts from article title"))
    p.add_argument(
        "--title",
        required=True,
        help="Article title (will be embedded into prompts)",
    )
    p.add_argument(
        "--out",
        required=False,
        help=("Optional output path (if omitted will write to " "docs/nano_banana_prompts_{slug}.txt)"),
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    prompts = build_prompts(args.title)

    out_path = args.out
    if not out_path:
        slug = slugify(args.title)
        out_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"nano_banana_prompts_{slug}.txt")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for line in prompts:
                f.write(line + "\n")
        print(f"Generated {len(prompts)} prompts -> {out_path}")
    except Exception as e:
        print(f"Error writing prompts file: {e}")
        for line in prompts:
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
