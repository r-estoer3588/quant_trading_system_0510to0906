<!-- docs/branding/articles/first_note_article.md -->

# Playwright × 生成 AI で“落ちない E2E”を 1 週間で立ち上げる（実務テンプレ付き）

最初の目標はシンプルです。「1 分で回る、壊れにくい核」を作ること。ここから少しずつ広げれば、CI での安定と運用コストの最小化が両立します。

この記事は“実務で動く最小セット”に絞り、すぐにコピペで流用できるテンプレを載せます。生成 AI（Copilot など）は“仕様化 → 削る”に使います。全体は物語的に、しかし手順は具体的に。

---

## 1. なぜ E2E は続かないのか — 修理しづらい設計

- 現象: グリーンからレッドへ、原因特定に時間がかかり、つい放置される。
- 根本: テストが“壊れたときの読みやすさ”を前提に設計されていない。
- 解決: 1 分で回る核 + 失敗時の証拠（スクショ/動画/ログ）が“差分で読める”こと。

## 2. 1 分で回る最小核（テンプレ）

playwright.config.ts（要点: 動画は失敗時のみ / retry で flake 吸収 / slowMo は局所で使う）

```ts
// playwright.config.ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  timeout: 30 * 1000,
  expect: { timeout: 5 * 1000 },
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: process.env.BASE_URL || "http://localhost:3000",
    trace: "on-first-retry",
    video: "retain-on-failure",
    screenshot: "only-on-failure",
    viewport: { width: 1280, height: 800 },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
```

最初のテスト（要点: locator ベース / `test.step` で読みやすく / 安定待ちの基本）

```ts
// tests/sanity.spec.ts
import { test, expect } from "@playwright/test";

test("トップの主要UIが表示される", async ({ page }) => {
  await test.step("トップへ", async () => {
    await page.goto("/");
    await expect(page).toHaveTitle(/My App|Sample/);
  });

  await test.step("ヘッダのリンク群", async () => {
    const nav = page.getByRole("navigation");
    await expect(nav).toBeVisible();
    await expect(nav.getByRole("link", { name: "ホーム" })).toBeVisible();
  });
});
```

ポイント

- `getByRole` / `getByTestId` を優先。CSS/XPath は最後の手段
- `expect(...).toBeVisible()` は“待ち”を内包するため積極活用
- `test.step` で失敗箇所を“物語の章立て”にする

## 3. 失敗は“差分”で読む — スクショ/動画/ログのセット

失敗時に何を見るかを先に決めておくと迷いが消えます。

- スクショ: 直前の画面と差分を見る（初手で“何が消え/増えたか”）
- 動画: タイミングのズレ（アニメ/トランジション）が原因のときに効く
- Trace: ネットワーク/コンソール/DOM スナップショット

実務メモ

- 一律 `waitForTimeout` は封印。`expect(...).toHave*` / `locator.waitFor` で待つ
- flake は retry 1〜2 回まで。上げすぎると“壊れたまま進む”

## 4. 生成 AI の使い方 — 仕様化 → 削る

Copilot などで“期待する UI 状態の箇条書き”を先に作り、そこからテストに落とします。

プロンプト例

```
アプリのトップ画面で、重要なUI要素の状態を10個、簡潔に列挙してください。
それぞれに、Playwrightのexpectで表現できる短い文（toBeVisible等）を添えてください。
```

出てきた 10 個から“今週の最小コア”として 3 つに削る。残りは backlog に回す。常に“読める失敗”を優先。

## 5. CI での安全弁（最低限）

- 1 回目 red → 2 回目 retry で green なら「flake」タグを付与、原因切り分けへ
- 連続 red は即 isolation（当該 spec のみ切り離して調査）
- 出力物（スクショ/動画/trace）は 7 日ローテーションで残す

## 6. 付録: コピペで使える断片

安定待ちの断片

```ts
await expect(page.getByRole("button", { name: "保存" })).toBeEnabled();
await page.getByRole("button", { name: "保存" }).click();
await expect(page.getByText("保存しました")).toBeVisible();
```

DOM 変化の差分を読む断片

```ts
const before = await page.locator("#count").innerText();
await page.getByRole("button", { name: "Add" }).click();
await expect(page.locator("#count")).toHaveText(String(Number(before) + 1));
```

スクショを任意に残す（調査用）

```ts
await page.screenshot({ path: `screenshots/${Date.now()}_state.png`, fullPage: true });
```

---

## まとめ

- まずは“1 分で回る核”を作り、壊れたときに“差分で読める”設計にする
- 生成 AI は仕様化に使い、テストは“削って”小さく始める
- CI は retry と隔離で“止まらない運用”を守る

次回は「失敗ログの読み方（スクショ ×DOM× ネットワークの三位一体）」を具体例で解説します。

---

付録: Alt テキスト例

> 背景は濃紺のグラデーション。左に「人とロボットの会話」イラスト、中央に『AI Narrative Studio』の文字。右側に淡い回路パターンと金色の小さな点が見える。
