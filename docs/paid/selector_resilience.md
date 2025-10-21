# セレクタ耐性の作り方（有料章）

この章では「なぜセレクタが壊れるのか」「どのようにして壊れにくいセレクタを設計するか」を、実務レベルで使えるテンプレと例を交えて解説します。最終的に得られるもの：

- すぐ使えるセレクタ設計テンプレ（`getByRole`, `getByTestId`, `locator` の組み合わせ）
- 失敗パターン別の修正手順（コピペで直せる差分）
- テスト側で DOM に安定的に `data-testid` を付与する最小コントラクト（アプリを壊さずにテストを安定化する方法）

---

## 1. 原因の整理（なぜ壊れるか）

よく見る原因は次の 4 つです（再掲）：

1. ロケール／フォント差による表示変化（表示テキストが変わる）
2. DOM 構造の変更（デザイナーが class を変えた）
3. 非同期描画のタイミング差（タイムアウトの使い方が原因）
4. 並列実行によるデータ競合（同一リソースを複数テストが操作）

これらを前提に「堅牢なセレクタ」を作るには、UI の意味（役割）で要素を参照する方針が有効です。

---

## 2. 優先順位：参照の強さ（推奨順）

1. getByRole（ARIA の役割に基づく検索）
2. getByTestId（`data-testid` 属性）
3. 見出しやラベルを使ったテキスト検索（正規表現で i18n を吸収）
4. CSS クラスや見た目に依存するセレクタ（最後の手段）

理由：ロールと testid は構造や見た目の変更に比較的強く、i18n やフォント差で壊れにくいからです。

---

## 3. 実践テンプレ（コピペで使える）

### 3.1 ボタンを押す（堅牢版）

```ts
// 優先：testid
const btnById = page.getByTestId("run-today-button");
if ((await btnById.count()) > 0) {
  await btnById.click();
  return;
}

// 次：ARIA role + 正規表現（ロケール差を吸収）
const btnByRole = page.getByRole("button", {
  name: /本日のシグナル|run today signals/i,
});
if ((await btnByRole.count()) > 0) {
  await btnByRole.first().click();
  return;
}

// 最後の手段：テキストベースの部分一致
await page.locator('button:has-text("本日のシグナル")').first().click();
```

### 3.2 テーブルを待つ

```ts
// 期待：st.dataframe (Streamlit) がある場合には data-testid を使う
await page.waitForSelector('[data-testid="stDataFrame"]', {
  state: "visible",
  timeout: 30000,
});

// 汎用：HTML table を待つ
await page.waitForSelector("table", { state: "visible", timeout: 30000 });
```

---

## 4. アプリを触らずに安定化する小技（テスト側での対応）

開発側に `data-testid` を追加してもらえれば一番よいのですが、すぐに協力を得られないことも多いです。その場合はテスト開始時に DOM に `data-testid` を注入する方法が現実的で安全です。

```ts
// e2e/helpers/annotateTestIds.ts 例
export async function addTestIds(page: Page) {
  await page.evaluate(() => {
    document.querySelectorAll("button").forEach((btn) => {
      const t = (btn.textContent || "").trim();
      if (/本日のシグナル|run today signals/i.test(t))
        btn.setAttribute("data-testid", "run-today-button");
    });
  });
}
```

利点:

- アプリコードを直接触らないためリスクが少ない
- テストの観点から最小限の ID だけを付与できる
- テストは `getByTestId()` を優先して使えるので安定化が早い

欠点:

- DOM の変化が激しい場合は注入ルールを都度更新する必要がある

---

## 5. 失敗時のデバッグ手順（ワークフロー）

1. まず `--trace on-first-retry` を有効にして再実行し、トレースとスクショを収集
2. `npx playwright show-trace` でトレースを開き、タイムアウト／待機箇所を解析
3. 問題が単純に待ち足りないだけなら `waitForSelector` に置換
4. セレクタの脆さが原因なら `data-testid` の注入 or `getByRole` の利用に切り替える

---

## 6. 付録：コピペ用チェックリスト（短縮版）

- [ ] `waitForTimeout` を検索して `waitForSelector` に置き換える候補を作る
- [ ] `getByRole` / `getByTestId` に書き換えられる箇所をチェック
- [ ] CI 実行時の `locale`, `timezone`, `viewport` を固定する
- [ ] 並列数を抑え、最終的に `workers: 1` で安定度を確認する

---

最後に: この章では「なぜそうするのか」を重視しました。単純に修正を適用するだけでなく、どの発想でその修正が有効なのかを理解すると、類似の壊れ方に対しても速やかに対処できるようになります。
