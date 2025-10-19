from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import urllib.request
import tempfile


def make_checklist(path: str) -> None:
    # Ensure a Japanese-capable TTF is registered with reportlab
    font_name = "NotoSansCJK"
    registered = False
    for candidate in [
        # common Windows fonts
        r"C:\\Windows\\Fonts\\meiryo.ttc",
        r"C:\\Windows\\Fonts\\meiryo.ttf",
        r"C:\\Windows\\Fonts\\YuGothR.ttc",
        # common Noto path if present
        r"/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if os.path.exists(candidate):
            try:
                pdfmetrics.registerFont(TTFont(font_name, candidate))
                registered = True
                break
            except Exception:
                pass

    # If not found, download a Noto CJK package to a temp file and register
    if not registered:
        try:
            tmpdir = tempfile.gettempdir()
            ttf_path = os.path.join(tmpdir, "NotoSansCJKjp-Regular.otf")
            if not os.path.exists(ttf_path):
                url = (
                    "https://noto-website-2.storage.googleapis.com/pkgs/"
                    "NotoSansCJKjp-hinted.zip"
                )
                zip_path = os.path.join(tmpdir, "noto_jp.zip")
                urllib.request.urlretrieve(url, zip_path)
                import zipfile

                with zipfile.ZipFile(zip_path) as z:
                    # extract a common OTF/TTF inside
                    for name in z.namelist():
                        if name.lower().endswith((".otf", ".ttf")):
                            z.extract(name, tmpdir)
                            extracted = os.path.join(tmpdir, name)
                            os.rename(extracted, ttf_path)
                            break
            pdfmetrics.registerFont(TTFont(font_name, ttf_path))
            registered = True
        except Exception:
            # fallback to Helvetica (will not display Japanese correctly)
            font_name = "Helvetica"

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    margin = 20 * mm

    # Header
    c.setFont(font_name, 18)
    c.drawString(margin, height - margin, "Week2: Playwright CI 整備チェックリスト")

    c.setFont(font_name, 10)
    y = height - margin - 18 - 8

    intro = (
        "このチェックリストは、Playwright を使った E2E テストの CI でよく起きる失敗を減らすための"
        "実用項目です。まず上から順に確認し、必要箇所を修正してください。"
    )
    text_width = width - margin * 2
    from reportlab.lib.utils import simpleSplit

    lines = simpleSplit(intro, "Helvetica", 9, text_width)
    for ln in lines:
        c.drawString(margin, y, ln)
        y -= 12

    y -= 6

    checklist = [
        "固定待機(waitForTimeout)を条件待機(waitForSelector/getByRole等)に置き換えた",
        "主要なセレクタを getByRole / data-testid に置き換え、UI 依存を減らした",
        "playwright.config.ts で viewport/locale/timezone を固定した",
        "CI コンテナに必要なフォントをインストールする Dockerfile を用意した",
        "スクリーンショット比較の閾値と差分ポリシーを定めた",
        "並列実行による共有リソース競合を検出し、該当テストを serial に分離した",
        "テストごとに一意のテストデータ（UUID サフィックス等）を導入した",
        "外部 API はモック/フェイクを利用するか、安定版のステージ環境を用意した",
        "重要な失敗時のログ/スクショ収集テンプレを導入した",
        "CI 実行設定とローカル環境の依存バージョンをドキュメント化した",
    ]

    c.setFont(font_name, 12)
    c.drawString(margin, y, "チェックリスト")
    y -= 18

    c.setFont(font_name, 11)
    for i, item in enumerate(checklist, start=1):
        # wrap text if necessary
        item_lines = simpleSplit(f"{i}. {item}", "Helvetica", 11, text_width)
        for ln in item_lines:
            c.drawString(margin + 6, y, ln)
            y -= 14
        y -= 4
        if y < 40 * mm:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 11)

    # Footer / usage note
    if y < 60 * mm:
        c.showPage()
        y = height - margin

    c.setFont(font_name, 9)
    c.drawString(
        margin,
        30 * mm,
        "Generated checklist — adapt to your CI environment before use.",
    )

    c.save()


if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "docs",
        "downloads",
        "checklist_week2.pdf",
    )
    make_checklist(out)
    print("Wrote:", out)
