# 英語版アイキャッチ画像 - 文言リスト

**目的**: 海外流入を狙い、Week1 メイン記事公開前（10 月 17 日まで）に英語版アイキャッチを準備。

---

## Week 1: Copilot Chat 10 選

### 日本語版（既存）

- **タイトル**: 対話で学ぶ Playwright × AI
- **副題**: E2E テスト自動化入門 - 初心者から CI まで 5 分で完結
- **ALT テキスト**: 左にノート PC の若手エンジニア（？の吹き出し）、右にメンター（！の吹き出し）。中央に「対話で学ぶ Playwright × AI」。濃紺の背景と右側のグリッド。

### 英語版（Week1 用 - NEW）

- **タイトル**: 10 Hidden Features of GitHub Copilot Chat
- **副題**: Supercharge Testing, Refactoring & Writing with AI
- **ALT テキスト**: Left: junior dev with laptop (? speech bubble), Right: mentor (! speech bubble). Center: "10 Hidden Features of Copilot Chat". Navy background with right-side grid.

### 生成コマンド例（Week1 用）

既存の人物 PNG（ユイ/レン）を再利用する場合:

```powershell
C:\Repos\quant_trading_system\venv\Scripts\python.exe tools\generate_article_eyecatch.py `
  --yui "C:\Users\stair\Downloads\yui.png" `
  --ren "C:\Users\stair\Downloads\ren.png" `
  --out "C:\Users\stair\Downloads\eyecatch_copilot_10_en.png" `
  --title1 "10 Hidden Features" `
  --title2 "of GitHub Copilot Chat" `
  --unify-style --posterize-bits 4 --saturation 0.95 --contrast 1.05 --tint navy --stroke 2
```

人物なし（タイトルのみ）の場合:

```powershell
C:\Repos\quant_trading_system\venv\Scripts\python.exe tools\generate_article_eyecatch.py `
  --out "C:\Users\stair\Downloads\eyecatch_copilot_10_en_simple.png" `
  --title1 "10 Hidden Features" `
  --title2 "of GitHub Copilot Chat" `
  --unify-style --posterize-bits 4 --saturation 0.95 --contrast 1.05 --tint navy
```

---

## Week 2: CI 失敗を 30 分で潰す

### 日本語版

- **タイトル**: 対話で学ぶ Playwright の CI 失敗を 30 分で潰す
- **副題**: タイムアウト・セレクタ・環境差異を対話形式で解決

### 英語版

- **タイトル**: Fixing Playwright CI Failures in 30 Minutes
- **副題**: Solve Timeouts, Selectors & Environment Issues
- **ALT テキスト**: Left: junior dev with laptop (? speech bubble), Right: mentor (! speech bubble). Center: "Fixing CI Failures in 30 Min". Navy background with right-side grid.

### 生成コマンド例（Week2 用）

```powershell
C:\Repos\quant_trading_system\venv\Scripts\python.exe tools\generate_article_eyecatch.py `
  --yui "C:\Users\stair\Downloads\yui.png" `
  --ren "C:\Users\stair\Downloads\ren.png" `
  --out "C:\Users\stair\Downloads\eyecatch_ci_failures_en.png" `
  --title1 "Fixing CI Failures" `
  --title2 "in 30 Minutes" `
  --unify-style --posterize-bits 4 --saturation 0.95 --contrast 1.05 --tint teal --stroke 2
```

---

## Week 3: 落ちないテスト設計

### 日本語版

- **タイトル**: 対話で学ぶ Playwright E2E テストを「落ちないテスト」にする 3 つの設計
- **副題**: 待機戦略・リトライ設定・テスト隔離を対話で理解

### 英語版

- **タイトル**: 3 Strategies for Stable E2E Tests
- **副題**: Waiting, Retry & Isolation in Playwright
- **ALT テキスト**: Left: junior dev with laptop (? speech bubble), Right: mentor (! speech bubble). Center: "3 Strategies for Stable E2E Tests". Navy background with right-side grid.

### 生成コマンド例（Week3 用）

```powershell
C:\Repos\quant_trading_system\venv\Scripts\python.exe tools\generate_article_eyecatch.py `
  --yui "C:\Users\stair\Downloads\yui.png" `
  --ren "C:\Users\stair\Downloads\ren.png" `
  --out "C:\Users\stair\Downloads\eyecatch_stable_e2e_en.png" `
  --title1 "3 Strategies for" `
  --title2 "Stable E2E Tests" `
  --unify-style --posterize-bits 4 --saturation 0.95 --contrast 1.05 --tint gold --stroke 2
```

---

## Week 4: AI×E2E の運用設計

### 日本語版

- **タイトル**: 対話で学ぶ AI 時代の E2E テスト運用設計
- **副題**: Copilot × Playwright でチーム全体の品質を上げる仕組み

### 英語版

- **タイトル**: E2E Test Operations in the AI Era
- **副題**: Team Quality with Copilot × Playwright
- **ALT テキスト**: Left: junior dev with laptop (? speech bubble), Right: mentor (! speech bubble). Center: "E2E Operations in AI Era". Navy background with right-side grid.

### 生成コマンド例（Week4 用）

```powershell
C:\Repos\quant_trading_system\venv\Scripts\python.exe tools\generate_article_eyecatch.py `
  --yui "C:\Users\stair\Downloads\yui.png" `
  --ren "C:\Users\stair\Downloads\ren.png" `
  --out "C:\Users\stair\Downloads\eyecatch_e2e_ops_ai_en.png" `
  --title1 "E2E Operations" `
  --title2 "in AI Era" `
  --unify-style --posterize-bits 4 --saturation 0.95 --contrast 1.05 --tint navy --stroke 2
```

---

## カラーティント（Tint）の使い分け

各週で異なるティントを使うことで、シリーズ全体に統一感を持たせつつ差別化:

- **Week1**: `--tint navy`（濃紺 - 落ち着いた信頼感）
- **Week2**: `--tint teal`（ティール - 実践的な緊張感）
- **Week3**: `--tint gold`（ゴールド - 安定感・成功イメージ）
- **Week4**: `--tint navy`（濃紺 - 総まとめの落ち着き）

---

## ALT テキストの共通パターン

すべての週で以下の構造を維持:

```
Left: junior dev with laptop (? speech bubble),
Right: mentor (! speech bubble).
Center: "[記事タイトル]".
Navy background with right-side grid.
```

これにより、シリーズ全体の視覚的一貫性とアクセシビリティを確保。

---

## 生成時の注意点

1. **人物 PNG の再利用**: 既存の透過 PNG（ユイ/レン）を使い回すことで、キャラクターの一貫性を保つ
2. **タイトルの改行**: 英語は日本語より長くなりがちなので、`--title1` と `--title2` で適切に改行
3. **posterize-bits**: 4-6 の範囲で調整し、イラスト風のタッチを維持
4. **stroke（白フチ）**: 2-3 の範囲で調整し、背景との視認性を確保
5. **出力先**: `C:\Users\stair\Downloads\` に統一し、ファイル名は `eyecatch_[記事略称]_en.png` の形式

---

## 次のアクション

- [ ] Week1 用の英語版アイキャッチを 10 月 17 日までに生成
- [ ] note 記事の画像差し替え（公開前）
- [ ] 英語版 ALT テキストを note に設定
- [ ] Week2-4 の英語版アイキャッチは各週の公開直前に生成

これで海外からの流入も狙えます 🌏
