# E2E Tests with Playwright

このディレクトリには、Streamlit UI の End-to-End テストが含まれています。

## 🎯 目的

- Streamlit UI の自動動作確認
- シグナル表示の正確性検証
- 手動確認作業の自動化

## 📁 構成

```
e2e/
├── streamlit/
│   ├── app-basic.spec.ts        # 基本動作テスト
│   ├── signals-display.spec.ts  # シグナル表示テスト
│   └── allocation-display.spec.ts # 配分結果テスト
└── example.spec.ts              # Playwright サンプル（参考用）
```

## 🚀 実行方法

### 1. 基本実行（ヘッドレス）

```powershell
npm test
```

### 2. ブラウザ表示付き実行

```powershell
npm run test:headed
```

### 3. インタラクティブ UI モード

```powershell
npm run test:ui
```

### 4. Streamlit テストのみ実行

```powershell
npm run test:streamlit
```

### 5. テスト結果レポート表示

```powershell
npm run test:report
```

## 🔧 前提条件

### 自動起動モード（推奨）

テスト実行時に Streamlit が自動で起動します（`playwright.config.ts` で設定済み）。

### 手動起動モード

既に Streamlit が起動している場合:

```powershell
# Python 仮想環境をアクティベート
.\venv\Scripts\Activate.ps1

# Streamlit を起動
streamlit run apps/app_integrated.py

# 別のターミナルでテスト実行
npm test
```

## 📊 データ準備

テストを実行する前に、シグナルデータを生成しておくと、より実践的なテストが可能です:

```powershell
# 当日シグナルを生成
python scripts/run_all_systems_today.py --parallel --save-csv
```

## 🎨 テストのカスタマイズ

### 特定のテストファイルのみ実行

```powershell
npx playwright test e2e/streamlit/app-basic.spec.ts
```

### 特定のブラウザで実行

```powershell
npx playwright test --project=chromium
```

### デバッグモード

```powershell
npx playwright test --debug
```

## 📝 テスト追加のガイドライン

新しいテストを追加する場合:

1. `e2e/streamlit/` 配下に `*.spec.ts` ファイルを作成
2. `test.describe()` でテストグループを定義
3. `test()` で個別のテストケースを記述
4. Streamlit の要素は `[data-testid="..."]` で取得

### テンプレート

```typescript
import { test, expect } from "@playwright/test";

test.describe("新機能のテスト", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");
  });

  test("機能が正しく動作する", async ({ page }) => {
    // テストロジック
  });
});
```

## 🔍 トラブルシューティング

### Streamlit が起動しない

```powershell
# Python 環境を確認
python --version

# Streamlit がインストールされているか確認
pip show streamlit
```

### テストがタイムアウトする

`playwright.config.ts` の `timeout` を調整:

```typescript
use: {
  actionTimeout: 30 * 1000, // 30秒
},
```

### スクリーンショットで原因を調査

失敗したテストのスクリーンショットは `test-results/` に自動保存されます。

## 📚 参考リンク

- [Playwright 公式ドキュメント](https://playwright.dev/)
- [Streamlit テスト戦略](https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-test-your-streamlit-app)
