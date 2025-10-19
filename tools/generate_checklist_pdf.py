from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def make_checklist(path: str):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    margin = 20 * mm

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, height - margin, "Week2: Playwright CI 整備チェックリスト")

    c.setFont("Helvetica", 10)
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

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "チェックリスト")
    y -= 18

    c.setFont("Helvetica", 11)
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

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(margin, 30 * mm, "Generated checklist — adapt to your CI environment before use.")

    c.save()


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "downloads", "checklist_week2.pdf")
    make_checklist(out)
    print("Wrote:", out)
