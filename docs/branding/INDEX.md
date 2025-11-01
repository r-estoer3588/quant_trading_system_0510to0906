<!-- docs/branding/INDEX.md -->

# AI Narrative Studio — Branding Kit

このフォルダは、X / note 運用をすぐ開始できる最低限の実務キットです。

- 目的: 初回セットアップの迷いをゼロにする（プロフィール文、固定ポスト、投稿計画、記事アウトライン）
- 生成系スクリプト: `tools/generate_header.py`, `tools/process_icon.py`

## アセットの現在地

- X ヘッダー: `c:\Users\stair\Downloads\header_twitter_1500x500_lefticon_overlay.png`
- note ヘッダー: `c:\Users\stair\Downloads\header_note_1280x670_lefticon_overlay.png`
- プロフィールアイコン（推奨）:
  - X: `c:\Users\stair\Downloads\processed_icon_latest\icon_circle_400.png`
  - note: `c:\Users\stair\Downloads\processed_icon_latest\icon_circle_512.png`

生成し直す場合は、`tools/generate_header.py` を使います（引数例は各ファイル内にコメント済み）。

## ブランド・ガイド（簡易）

- カラー
  - ダークブルー: #142A4D 前後
  - ブルー: #2C507A 前後
  - アクセントグリーン: #00C896 前後（回路パターン）
  - ゴールド: #FFC832 前後（小ドット）
- フォント候補（Windows 標準 → 自動フォールバック）
  - Segoe UI / Arial / Calibri
- モチーフ
  - 「人間 × ロボットの対話」＋「回路パターン（知の流れ）」＋「ゴールドで温度感」

## コンテンツの柱（3 本）

1. QA 自動化とテスト運用（Playwright / Copilot / 実務 Tips）
2. 生成 AI を使った業務効率化（手順とテンプレを具体で）
3. クオンツ開発の学び共有（失敗学と検証姿勢）

この 3 本の柱から週ごとに均等配分する方針（詳細は `content_calendar.md`）。

---

- プロフィール文: `bios.md`
- 固定ポスト: `pinned_tweet.md`
- 投稿計画: `content_calendar.md`
- 初回記事アウトライン: `article_playwright_ai_outline.md`
- テンプレ集（投稿/スレッド/Alt テキスト）: `templates.md`
