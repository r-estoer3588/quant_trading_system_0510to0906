Playwright mini reproducible example

このミニリポジトリは、よくあるフレーキーテスト（固定待機時間の誤使用）を示す `fail` と、
条件待機に置き換えた `fix` を対比するための最小構成です。

使い方:

1. Node.js をインストール
2. cd repro/playwright-mini
3. npm ci
4. npx playwright test

構成:

- `tests/fail.spec.ts` - 固定待機を使って失敗するパターン
- `tests/fix.spec.ts` - `waitForSelector` を使って安定させたパターン
- `playwright.config.ts` - ミニリポジトリ用の設定

記事付録では、このディレクトリを `docs/playwright_mini_repro.zip` にアーカイブして付属します。
