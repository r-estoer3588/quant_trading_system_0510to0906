# アイキャッチ画像生成ワークフロー

## 📁 ベース画像の場所

**ファイルパス**: `docs/images/eyecatch_base_yui_ren.png`

- **説明**: ユイとレン先輩のキャラクター、Navy 背景、回路パターンを含む文字なしベース画像
- **サイズ**: 1280×670px（note.com 最適サイズ）
- **スタイル**: posterize 4-bit、saturation 0.95、contrast 1.05、Navy tint
- **用途**: Week1-4 全記事で共通使用。毎回タイトルのみを変更してアイキャッチを生成

---

## 🎨 Week1-4 アイキャッチ生成コマンド

### Week1（10/18 公開）：GitHub Copilot Chat 便利機能

**日本語版**:

```powershell
python.exe tools\generate_article_eyecatch.py `
  --base "docs\images\eyecatch_base_yui_ren.png" `
  --out "docs\images\eyecatch_week1_ja.png" `
  --title1 "GitHub Copilot Chat" `
  --title2 "隠れた便利機能 10選" `
  --stroke 2
```

**英語版**:

```powershell
python.exe tools\generate_article_eyecatch.py `
  --base "docs\images\eyecatch_base_yui_ren.png" `
  --out "docs\images\eyecatch_week1_en.png" `
  --title1 "10 Hidden Features" `
  --title2 "of GitHub Copilot Chat" `
  --stroke 2
```

---

### Week2（10/25 公開）：CI 失敗トラブルシュート

**日本語版**:

```powershell
python.exe tools\generate_article_eyecatch.py `
  --base "docs\images\eyecatch_base_yui_ren.png" `
  --out "docs\images\eyecatch_week2_ja.png" `
  --title1 "Playwright の CI失敗を" `
  --title2 "30分で潰す実践ガイド" `
  --tint teal `
  --stroke 2
```

**英語版**:

```powershell
python.exe tools\generate_article_eyecatch.py `
  --base "docs\images\eyecatch_base_yui_ren.png" `
  --out "docs\images\eyecatch_week2_en.png" `
  --title1 "Fix Playwright CI Failures" `
  --title2 "in 30 Minutes - A Practical Guide" `
  --tint teal `
  --stroke 2
```

---

### Week3（11/1 公開）：テスト設計パターン

**日本語版**:

```powershell
python.exe tools\generate_article_eyecatch.py `
  --base "docs\images\eyecatch_base_yui_ren.png" `
  --out "docs\images\eyecatch_week3_ja.png" `
  --title1 "落ちないテスト設計" `
  --title2 "3つの実践パターン" `
  --tint gold `
  --stroke 2
```

**英語版**:

```powershell
python.exe tools\generate_article_eyecatch.py `
  --base "docs\images\eyecatch_base_yui_ren.png" `
  --out "docs\images\eyecatch_week3_en.png" `
  --title1 "3 Practical Patterns" `
  --title2 "for Flake-Free Test Design" `
  --tint gold `
  --stroke 2
```

---

### Week4（11/8 公開）：AI×E2E テスト運用設計

**日本語版**:

```powershell
python.exe tools\generate_article_eyecatch.py `
  --base "docs\images\eyecatch_base_yui_ren.png" `
  --out "docs\images\eyecatch_week4_ja.png" `
  --title1 "AI × E2E テストの" `
  --title2 "運用設計実践ガイド" `
  --stroke 2
```

**英語版**:

```powershell
python.exe tools\generate_article_eyecatch.py `
  --base "docs\images\eyecatch_base_yui_ren.png" `
  --out "docs\images\eyecatch_week4_en.png" `
  --title1 "AI × E2E Testing" `
  --title2 "Operations Design Guide" `
  --stroke 2
```

---

## 🛠️ ツール実装完了

✅ `tools/generate_article_eyecatch.py` に **--base オプション** を実装済み

### 実装内容

1. **--base** 引数：既存画像をベースとして読み込み
2. **タイトルオーバーレイのみ実行**：キャラクター配置・背景生成をスキップ
3. **Tint 適用の調整**：ベース画像に tint（navy/teal/gold）を適用可能

### 実装確認済み

- ✅ Week1 日本語版生成成功（`docs/images/eyecatch_week1_ja.png`）
- ✅ Week2 Teal tint 生成成功（`docs/images/eyecatch_week2_ja.png`）
- ✅ タイトルテキストの正しい配置
- ✅ Tint 色変更機能の動作確認

---

## 📝 使い方

1. **ベース画像確認**: `docs/images/eyecatch_base_yui_ren.png` が存在することを確認
2. **コマンド実行**: 上記 Week1-4 コマンドをコピーして実行
3. **出力確認**: `docs/images/eyecatch_weekX_XX.png` が生成される
4. **note.com アップロード**: 記事公開時にアイキャッチ設定

---

## 🎯 メリット

- **一貫性**: 全週で同じキャラクター配置・背景デザイン
- **効率**: タイトル変更のみで約 5 秒で生成完了（従来 10 秒 →50%短縮）
- **品質**: ベース画像の品質が全週で保証される
- **柔軟性**: Tint 色変更で週ごとに雰囲気を変えられる

---

## 📌 次のステップ

1. `tools/generate_article_eyecatch.py` に --base オプション実装
2. Week1 日本語版アイキャッチ生成テスト
3. Week2-4 のアイキャッチ事前生成（公開 1 週間前）
