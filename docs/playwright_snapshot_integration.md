Playwright snapshot integration (UI metrics)

目的

- Streamlit の UI で出力される `results_csv/ui_metrics_*.json` を Playwright のテスト後に比較し、期待値（baseline）と異なる場合に CI を失敗させるための手順例を示します。

要点

- スナップショットは `apps/app_today_signals.py` の `_export_metrics_snapshot()` によって `results_csv/ui_metrics_YYYYMMDD_HHMMSS.json` として保存されます。
- 比較は `tools/compare_ui_metrics.py` を使い、数値丸めやキー整列による正規化を行って差分を検出します。
- CI では `tools/playwright_snapshot_check.py` を実行して差分があれば非ゼロ終了コードを返すようにします。

CI ワークフロー（例）

```yaml
# .github/workflows/ui_snapshot_check.yml
name: UI snapshot check
on: [workflow_run]
jobs:
  ui-snapshot:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run Playwright tests (example)
        run: |
          # ここで Playwright のテストを実行し、Streamlit を操作して UI を描画・スナップショットを作らせる
          # 省略: プロジェクト固有の Playwright 実行コマンドをここに書く
          npm run test:ui || true
      - name: Compare UI snapshot
        run: |
          # baseline はリポジトリに含めておく or アーティファクトから取得
          python tools/playwright_snapshot_check.py --baseline results_csv/ui_metrics_baseline.json
```

Playwright テスト内での呼び出し例（TypeScript）

```ts
// e2e/playwright_capture_and_compare.spec.ts
import { test, expect } from "@playwright/test";
import { execSync } from "child_process";
import fs from "fs";

test("capture ui metrics and compare", async ({ page }) => {
  // Streamlit アプリの URL を指定
  await page.goto("http://localhost:8501");

  // 必要な UI 操作を行い、アプリが実行されるのを待つ。
  // Playwright テストはメインの実行ボタン「Generate Signals」を押して
  // パイプラインを走らせ、完了メッセージ ("Signals generation complete") を待ちます。
  const runButton = page.locator("text=Generate Signals");
  if ((await runButton.count()) > 0) {
    await runButton.click();
    await page.waitForSelector("text=Signals generation complete", {
      timeout: 120_000,
    });
  } else {
    await page.waitForTimeout(2000);
  }

  // UI がエクスポートした最新の snapshot ファイルを探して比較する
  // Playwright から Python スクリプトを直接呼ぶ例
  try {
    execSync(
      "python tools/playwright_snapshot_check.py --baseline results_csv/ui_metrics_baseline.json",
      { stdio: "inherit" }
    );
  } catch (e) {
    // 失敗時は test を失敗させる
    throw e;
  }
});
```

安定化のヒント

- スナップショットは数値を丸め（compare スクリプトでデフォルト 6 桁）、キー順で比較するようにしています。必要なら `--round` オプションで丸め桁数を調整してください。
- 実行環境で時刻や外部データに依存する場合、Playwright 側で固定データ（モック）や事前キャッシュ更新を行ってからスナップショットを取得すると安定します。

必要なら、このワークフローを `.github/workflows/` に追加するための具体的な YAML を作成してコミットします。
