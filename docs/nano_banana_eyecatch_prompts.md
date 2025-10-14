# nano banana アイキャッチ生成プロンプト集

**目的**: ベース画像（`docs/images/eyecatch_base_yui_ren.png`）にタイトルテキストを追加

**前提**: ベース画像には既にユイ&レン先輩、Navy 背景、回路パターンが含まれている

**使い方**:

1. ベース画像を nano banana にアップロード
2. 各 Week のプロンプトをコピー
3. 生成 → `docs/images/eyecatch_weekX_XX.png` に保存

---

## 🎨 基本方針

- **ベース画像**: `docs/images/eyecatch_base_yui_ren.png` を使用
- **追加要素**: タイトルテキストのみ（キャラクター・背景は既にある）
- **タイトル位置**: 画面上部（上から 15-20%、キャラクターと重ならない位置）
- **フォント**: 太字、読みやすいゴシック体
- **色**: 1 行目=白、2 行目=ゴールド（#FFDC64）
- **影**: 黒いドロップシャドウ

---

## 📋 Week1: GitHub Copilot Chat（10/18 公開）

### 日本語版

```
この画像にタイトルテキストを追加してください。

【タイトル】
- 位置: 画面上部（上から15-20%）
- 1行目: 「GitHub Copilot Chat」（白文字、太字、大きめ）
- 2行目: 「隠れた便利機能 10選」（ゴールド #FFDC64、1行目より小さめ）
- テキストに黒いドロップシャドウ追加
- キャラクターと重ならないように配置
- 必要なら半透明の黒背景ボックスを追加
```

### 英語版

```
Add title text to this image.

【Title】
- Position: Upper area (15-20% from top)
- Line 1: "10 Hidden Features" (white, bold, larger)
- Line 2: "of GitHub Copilot Chat" (gold #FFDC64, smaller)
- Add black drop shadow
- Avoid overlapping with characters
- Optional: add semi-transparent background box
```

---

## 📋 Week2: Playwright CI 失敗（10/25 公開）

### 日本語版

```
この画像にティールグリーンの色合いとタイトルテキストを追加してください。

【色合い】
- 画像全体にティールグリーン（#00C896）のオーバーレイ（透明度30%）

【タイトル】
- 位置: 画面上部（上から15-20%）
- 1行目: 「Playwright の CI失敗を」（白文字、太字、大きめ）
- 2行目: 「30分で潰す実践ガイド」（ゴールド #FFDC64、1行目より小さめ）
- テキストに黒いドロップシャドウ追加
- キャラクターと重ならないように配置
- 必要なら半透明の黒背景ボックスを追加
```

### 英語版

```
Add teal green tint and title text to this image.

【Color】
- Add teal green (#00C896) overlay to entire image (30% opacity)

【Title】
- Position: Upper area (15-20% from top)
- Line 1: "Fix Playwright CI Failures" (white, bold, larger)
- Line 2: "in 30 Minutes" (gold #FFDC64, smaller)
- Add black drop shadow
- Avoid overlapping with characters
- Optional: add semi-transparent background box
```

---

## 📋 Week3: テスト設計（11/1 公開）

### 日本語版

```
この画像にゴールドの色合いとタイトルテキストを追加してください。

【色合い】
- 画像全体にゴールド（#FFC832）のオーバーレイ（透明度20%）

【タイトル】
- 位置: 画面上部（上から15-20%）
- 1行目: 「落ちないテスト設計」（白文字、太字、大きめ）
- 2行目: 「3つの実践パターン」（ゴールド #FFDC64、1行目より小さめ）
- テキストに黒いドロップシャドウ追加
- キャラクターと重ならないように配置
- 必要なら半透明の黒背景ボックスを追加
```

### 英語版

```
Add gold tint and title text to this image.

【Color】
- Add gold (#FFC832) overlay to entire image (20% opacity)

【Title】
- Position: Upper area (15-20% from top)
- Line 1: "3 Practical Patterns" (white, bold, larger)
- Line 2: "for Flake-Free Test Design" (gold #FFDC64, smaller)
- Add black drop shadow
- Avoid overlapping with characters
- Optional: add semi-transparent background box
```

---

## 📋 Week4: AI×E2E（11/8 公開）

### 日本語版

```
この画像にタイトルテキストを追加してください。

【タイトル】
- 位置: 画面上部（上から15-20%）
- 1行目: 「AI × E2E テストの」（白文字、太字、大きめ）
- 2行目: 「運用設計実践ガイド」（ゴールド #FFDC64、1行目より小さめ）
- テキストに黒いドロップシャドウ追加
- キャラクターと重ならないように配置
- 必要なら半透明の黒背景ボックスを追加
```

### 英語版

```
Add title text to this image.

【Title】
- Position: Upper area (15-20% from top)
- Line 1: "AI × E2E Testing" (white, bold, larger)
- Line 2: "Operations Design Guide" (gold #FFDC64, smaller)
- Add black drop shadow
- Avoid overlapping with characters
- Optional: add semi-transparent background box
```

---

## 🎯 色の使い分け

| Week | テーマ       | 色合い追加                 | 狙い             |
| ---- | ------------ | -------------------------- | ---------------- |
| 1    | Copilot 機能 | なし（Navy 背景そのまま）  | シリーズの顔     |
| 2    | CI 失敗解決  | Teal #00C896（透明度 30%） | 問題解決         |
| 3    | テスト設計   | Gold #FFC832（透明度 20%） | 信頼性・安定性   |
| 4    | AI×E2E 運用  | なし（Navy 背景そのまま）  | 原点回帰・完結編 |

---

## 💡 使い方のコツ

### 1. ベース画像アップロード

nano banana で「画像編集」を選択 → ベース画像をアップロード

### 2. プロンプト貼り付け

該当 Week のプロンプトをそのままコピペ

### 3. 微調整（必要時）

```
タイトルをもう少し上に（上から10%の位置に）
```

```
背景ボックスを追加して、もっと読みやすく
```

### 4. 保存

生成された画像を `docs/images/eyecatch_weekX_XX.png` に保存

---

## 📌 次のアクション

1. **Week1 生成**: 上記プロンプトで nano banana 実行
2. **確認**: 文字がキャラクターと重なっていないか確認
3. **調整**: 必要なら位置・サイズ・背景を微調整
4. **保存**: `eyecatch_week1_ja.png` に保存

Week1 公開まであと 3 日！🚀
