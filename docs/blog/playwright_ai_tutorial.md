# 【対話で学ぶ】Playwright × AI で E2E テスト自動化入門

> **✨ AI Narrative Studio へようこそ！**  
> このシリーズでは、AI と人間の「対話」を通じて、技術を物語のように学んでいきます。  
> 今回は、Web アプリのテスト自動化ツール「Playwright」を、AI を活用しながら 1 日で使いこなす方法をお届けします。

---

> **📖 この記事で分かること**
>
> - Playwright とは何か、なぜ AI 時代に必須なのか
> - GitHub Copilot などの AI ツールと組み合わせた実践的な使い方
> - 初心者が陥りがちな罠とその解決策
> - CI/CD への組み込み方法

---

## 🎬 はじめに:テスト自動化の民主化時代が来た

Web アプリケーション開発において、「テストは書きたいけど時間がない」「テストコードの書き方が分からない」という悩みを抱えていませんか?

2025 年現在、**AI と Playwright の組み合わせ**によって、この問題は劇的に解決できるようになりました。プログラミング初心者でも、自然な言葉で「こういうテストがしたい」と伝えるだけで、AI が高品質なテストコードを生成してくれる時代です。

## 📖 この記事の登場人物

この記事では、開発経験 1 年の「**ユイ**」と、テスト自動化のベテラン「**レン先輩**」の対話を通じて、**Playwright の基礎から AI を活用した実践テクニック**まで、物語形式で分かりやすく解説します。

> 💡 **このシリーズの特徴**  
> 技術書のような堅苦しい説明ではなく、「会話」で学びます。実際の開発現場でのやり取りをイメージしながら読み進めてください。

---

## 📖 目次

1. 第 1 章：なぜ Playwright なのか
2. 第 2 章：最初の一歩を踏み出す
3. 第 3 章：実践的なテストケースの作成
4. 第 4 章：AI 協働の 3 つのパターン
5. 第 5 章：ビジュアルテストで見た目も守る
6. 第 6 章：CI/CD への組み込みで完全自動化
7. 第 7 章：よくあるつまずきポイントと解決策
8. 第 8 章：まとめ — AI × Playwright の未来

---

## 第 1 章：なぜ Playwright なのか

### 登場人物紹介

**ユイ（開発経験 1 年）**  
最近 GitHub Copilot を使い始めたエンジニア。テスト自動化に興味があるが、どこから手をつけていいか分からない。

**レン先輩（テスト自動化ベテラン）**  
親切で教え上手。実践的なアドバイスで後輩をサポート。

---

### 会話

**ユイ**: 先輩、最近 GitHub Copilot とか使ってコードを書いているんですけど、書いた Web アプリのテストはどうすればいいですか？手動で毎回ブラウザを開いてボタンを押すのが大変で…。

**レン**: いいところに気づいたね。そこで登場するのが**Playwright**だよ。これは「ブラウザを自動で操作してくれるツール」なんだ。

**ユイ**: ブラウザを自動で…？

**レン**: そう。たとえば「ログインボタンを押す」「フォームに文字を入力する」「結果が正しく表示されるか確認する」といった作業を、プログラムで書いておけば、あとは実行するだけで自動テストができる。

### Playwright の 3 つの強み

| 特徴                   | 説明                                                       |
| ---------------------- | ---------------------------------------------------------- |
| **クロスブラウザ対応** | Chrome、Firefox、Safari、Edge など主要ブラウザすべてに対応 |
| **高速・安定**         | 並列実行可能で、待機処理が賢く設計されている               |
| **開発者フレンドリー** | TypeScript/JavaScript/Python 対応、AI との相性 ◎           |

**ユイ**: それって、AI と組み合わせるとどうなるんですか？

**レン**: AI が「テストコードそのもの」を書いてくれるようになるんだ。君が「ログイン機能をテストしたい」と言えば、AI が Playwright のコードを生成してくれる。つまり、**テスト自動化の自動化**ができるようになる。

> 💡 **ポイント**  
> Playwright 単体でも強力ですが、AI と組み合わせることで「テストケースの設計」「コード生成」「デバッグ」まで全てが効率化されます。

---

## 第 2 章：最初の一歩を踏み出す

### ステップ 1：インストール

**ユイ**: 実際にやってみたいです！何から始めればいいですか？

**レン**: まずは Playwright をインストールしよう。Node.js を使っている場合はこうだね：

```bash
npm init playwright@latest
```

インストール時の質問は基本的に全て Enter で OK。すると以下が自動生成されます：

```
my-project/
├── tests/
│   └── example.spec.js
├── playwright.config.js
└── package.json
```

### ステップ 2：AI にテストコードを書いてもらう

**レン**: 手動で書かなくていい。AI に頼むんだ。

**ユイ**: どうやって？

**レン**: GitHub Copilot やチャット型 AI に、こう聞いてみて：

> 「Playwright で、`https://example.com`にアクセスして、ページタイトルが『Example Domain』であることを確認するテストコードを書いて」

すると、こんなコードが生成されます：

```javascript
// tests/example.spec.js
import { test, expect } from "@playwright/test";

test("ページタイトルを確認する", async ({ page }) => {
  // example.comにアクセス
  await page.goto("https://example.com");

  // タイトルが正しいか確認
  await expect(page).toHaveTitle("Example Domain");
});
```

### ステップ 3：実行してみる

```bash
npx playwright test
```

**ユイ**: 動きました！ブラウザが一瞬開いて、自動でページを確認して、テストが通りました！

```
Running 1 test using 1 worker
  ✓ example.spec.js:3:1 › ページタイトルを確認する (1.2s)

  1 passed (2s)
```

> 🎉 **成功体験ポイント**  
> わずか 5 分で最初のテストが動いた！この成功体験が重要です。

---

## 第 3 章：実践的なテストケースの作成

### ログイン機能のテスト

**ユイ**: でも実際の開発だと、もっと複雑なことをしますよね？たとえば、フォームに入力して送信するとか。

**レン**: そうだね。じゃあ、AI にこう聞いてみよう：

> 「Playwright で、ログインフォーム（メールアドレスとパスワード）に入力して、送信ボタンを押し、ダッシュボードに遷移することを確認するテストを書いて」

生成されるコード：

```javascript
// tests/login.spec.js
import { test, expect } from "@playwright/test";

test("ログイン機能のテスト", async ({ page }) => {
  // ログインページにアクセス
  await page.goto("https://yourapp.com/login");

  // メールアドレスを入力
  await page.fill('input[name="email"]', "test@example.com");

  // パスワードを入力
  await page.fill('input[name="password"]', "password123");

  // ログインボタンをクリック
  await page.click('button[type="submit"]');

  // ダッシュボードに遷移したか確認
  await expect(page).toHaveURL("https://yourapp.com/dashboard");

  // ダッシュボードに「ようこそ」と表示されているか確認
  await expect(page.locator("h1")).toContainText("ようこそ");
});
```

### セレクタが間違っていた場合の対処法

**ユイ**: でも、実際に動かしてみたら、入力フィールドのセレクタ（`input[name="email"]`とか）が違ってエラーになりました…。

**レン**: そういうときは**Playwright の便利機能**を使おう。`codegen`というツールがあって、これを使うと**実際にブラウザを操作しながら、自動でコードを生成してくれる**んだ。

```bash
npx playwright codegen https://yourapp.com/login
```

**ユイ**: やってみます…あ、ブラウザが開いて、操作すると右側にコードがリアルタイムで出てきました！

![Playwright Codegen](https://playwright.dev/img/codegen.png)

**レン**: そう。これで正確なセレクタが分かる。生成されたコードをコピーして、テストファイルに貼り付ければいい。

> 💡 **プロの Tips**  
> `codegen`は初心者の最強の味方。複雑な UI でも、実際に操作するだけで正確なコードが手に入ります。

---

## 第 4 章：AI 協働の 3 つのパターン

**レン**: Playwright を使うときの、AI との効果的な協働パターンを教えるね。

### パターン 1：テストケース設計を AI に相談

**AI への質問例**：

> 「EC サイトのカート機能をテストしたい。どんなテストケースが必要？」

**AI の回答例**：

```markdown
## カート機能のテストケース設計

### 基本機能

1. ✅ 商品をカートに追加できるか
2. ✅ カート内の数量を変更できるか
3. ✅ カートから商品を削除できるか

### 計算ロジック

4. ✅ 合計金額が正しく計算されるか
5. ✅ 複数商品の合計が正しいか
6. ✅ クーポン適用後の金額が正しいか

### エッジケース

7. ✅ 空のカートで購入ボタンが無効になるか
8. ✅ 在庫切れ商品がカートに入らないか
9. ✅ 上限数を超えて追加できないか
```

**ユイ**: なるほど！抜け漏れを防げますね。

### パターン 2：コード生成とレビュー

**ステップ 1**: AI にテストコード生成を依頼

**ステップ 2**: 生成されたコードを AI にレビューしてもらう

> 「この Playwright テストコードをレビューして。改善点や問題点があれば教えて」

**AI のレビュー結果例**：

```markdown
## コードレビュー結果

### ⚠️ 改善が必要な点

1. **待機処理が不足**

   - 問題：動的に読み込まれる要素に対して待機処理がない
   - 修正案：`await page.waitForSelector('.cart-item')`を追加

2. **エラーハンドリングが不十分**

   - 問題：ネットワークエラー時の挙動が未定義
   - 修正案：try-catch でラップし、スクリーンショットを保存

3. **テストデータのハードコーディング**
   - 問題：`test@example.com`が直接埋め込まれている
   - 修正案：環境変数から読み込む

### ✅ 良い点

- セレクタの選択が適切
- アサーションが明確
```

**ユイ**: 自動レビューしてくれるんですね！

### パターン 3：エラー解決のアシスタント

**エラー発生時**：

```
Error: Timeout 30000ms exceeded.
waiting for selector "button.submit" to be visible
```

**AI への質問**：

> 「Playwright でこのエラーが出ました。どう直せばいいですか？」

**AI の解決策**：

````markdown
## エラー解決策

### 原因の可能性

1. セレクタが間違っている
2. 要素が表示されるまで時間がかかっている
3. ページの読み込みが完了していない

### 解決方法

**方法 1: セレクタを確認**

```javascript
// デバッグ用に全ボタンを表示
const buttons = await page.locator("button").all();
console.log(await Promise.all(buttons.map((b) => b.getAttribute("class"))));
```
````

**方法 2: タイムアウトを延長**

```javascript
await page.waitForSelector("button.submit", { timeout: 60000 });
```

**方法 3: ネットワーク待機**

```javascript
await page.waitForLoadState("networkidle");
await page.click("button.submit");
```

**推奨**: 方法 1 で正しいセレクタを確認後、方法 3 を適用

````

---

## 第5章：ビジュアルテストで見た目も守る

### スクリーンショット比較テスト

**ユイ**: Playwrightって、見た目のテストもできるんですか？

**レン**: できるよ。**スクリーンショット比較**という機能がある。

```javascript
import { test, expect } from '@playwright/test';

test('トップページの見た目が変わっていないか', async ({ page }) => {
  await page.goto('https://yourapp.com');

  // スクリーンショットを撮影して比較
  await expect(page).toHaveScreenshot('homepage.png');
});
```

### 初回実行時

```bash
npx playwright test --update-snapshots
```

基準画像が `tests/__screenshots__/homepage.png` に保存される。

### 2 回目以降

```bash
npx playwright test
```

現在の画面と基準画像を比較。差分があればテスト失敗。

### 差分が検出された場合

```
Error: Screenshot comparison failed:
  Expected: tests/__screenshots__/homepage.png
  Received: tests/__screenshots__/homepage-actual.png
  Diff: tests/__screenshots__/homepage-diff.png
```

差分画像を確認して、意図的な変更なら：

```bash
npx playwright test --update-snapshots
```

> 🎨 **活用シーン**
>
> - CSS 変更時のレイアウト崩れ検出
> - デザインシステムの一貫性チェック
> - リファクタリング時の視覚的回帰テスト

---

## 第 6 章：CI/CD への組み込みで完全自動化

**ユイ**: これ、毎回手動で実行するんですか？

**レン**: いや、GitHub Actions などに組み込んで、**コードをプッシュしたら自動でテストが走る**ようにできる。

### AI に聞いてみる

> 「GitHub Actions で Playwright テストを実行するワークフローファイルを書いて」

### 生成されるワークフロー

```yaml
# .github/workflows/playwright.yml
name: Playwright Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: 18

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright browsers
        run: npx playwright install --with-deps

      - name: Run Playwright tests
        run: npx playwright test

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: playwright-report/

      - name: Upload screenshots on failure
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: screenshots
          path: test-results/
```

### 実行結果の確認

プルリクエストに自動でテスト結果が表示されます：

```
✅ All checks have passed
   Playwright Tests — 15 tests passed in 2m 34s
```

失敗時は：

```
❌ Some checks were not successful
   Playwright Tests — 2 of 15 tests failed
   📎 View test report and screenshots
```

**ユイ**: これでプルリクエストを出すたびに自動テストされるんですね！

---

## 第 7 章：よくあるつまずきポイントと解決策

### つまずき 1：「要素が見つからない」

**エラー例**：

```
Error: locator.click: Timeout 30000ms exceeded.
=========================== logs ===========================
waiting for locator('button#submit')
```

**AI に聞く**：

> 「Playwright で要素が見つからないとき、どうやってデバッグすればいいですか？」

**解決策**：

```javascript
// 方法1: 実行を一時停止して確認
await page.pause();

// 方法2: スクリーンショットを保存
await page.screenshot({ path: "debug.png", fullPage: true });

// 方法3: 要素の存在確認
const count = await page.locator("button#submit").count();
console.log(`Found ${count} elements`);

// 方法4: すべてのボタンを列挙
const buttons = await page.locator("button").all();
for (const btn of buttons) {
  console.log(await btn.textContent());
}
```

---

### つまずき 2：「テストが不安定（フレーキー）」

**症状**: たまに成功したり失敗したりする

**AI に聞く**：

> 「Playwright のテストが不安定です。どうすれば安定させられますか？」

**解決策**：

```javascript
// ❌ 不安定なコード
await page.click("button");
await expect(page.locator(".result")).toBeVisible();

// ✅ 安定したコード
await page.click("button");
await page.waitForLoadState("networkidle"); // 通信完了を待つ
await page.waitForSelector(".result", { state: "visible" }); // 要素が表示されるまで待つ
await expect(page.locator(".result")).toBeVisible();
```

**設定でリトライを有効化**：

```javascript
// playwright.config.js
export default {
  retries: process.env.CI ? 2 : 0, // CI環境では2回リトライ
  timeout: 60000, // タイムアウトを60秒に延長
};
```

---

### つまずき 3：「認証が必要なページのテスト」

**問題**: 毎回ログインするのは時間がかかる

**AI に聞く**：

> 「Playwright で、ログイン状態を保存して再利用する方法を教えて」

**解決策**：

```javascript
// tests/auth.setup.js
import { test as setup } from "@playwright/test";

const authFile = "playwright/.auth/user.json";

setup("authenticate", async ({ page }) => {
  // ログイン処理
  await page.goto("https://yourapp.com/login");
  await page.fill('input[name="email"]', "test@example.com");
  await page.fill('input[name="password"]', "password123");
  await page.click('button[type="submit"]');

  // ログイン状態を保存
  await page.context().storageState({ path: authFile });
});
```

```javascript
// playwright.config.js
export default {
  projects: [
    { name: "setup", testMatch: /.*\.setup\.js/ },
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        storageState: "playwright/.auth/user.json", // 保存した状態を読み込み
      },
      dependencies: ["setup"],
    },
  ],
};
```

これで、すべてのテストでログイン済み状態から開始できます！

---

## 第 8 章：まとめ — AI × Playwright の未来

### 今日学んだこと

✅ Playwright はブラウザ操作を自動化する最強ツール
✅ AI と組み合わせることで、テストコードの作成が劇的に効率化
✅ `codegen`ツールで正確なセレクタが簡単に取得できる
✅ CI/CD に組み込めば、完全自動テストが実現
✅ 困ったときは AI に聞けば、解決策が返ってくる

### AI × Playwright の 5 つの活用パターン

| パターン                   | 効果                                   |
| -------------------------- | -------------------------------------- |
| **テストケース設計の相談** | 抜け漏れを防ぎ、網羅的なテストを実現   |
| **コード生成**             | 自然な言葉から高品質なコードを自動生成 |
| **コードレビュー**         | ベストプラクティスに基づいた改善提案   |
| **エラー解決**             | エラーメッセージから解決策を即座に提示 |
| **CI/CD 設定**             | ワークフロー設定も AI が自動生成       |

### これからの学習ロードマップ

```mermaid
graph LR
    A[基本テスト作成] --> B[実践的なシナリオ]
    B --> C[ビジュアルテスト]
    C --> D[CI/CD統合]
    D --> E[パフォーマンステスト]
    E --> F[E2Eテスト完全自動化]
```

### 次のステップ

1. **今すぐ試す**: 自分のプロジェクトで 1 つテストを書いてみる
2. **習慣化**: 新機能を作ったら必ずテストも書く
3. **拡張**: API テスト、パフォーマンステストにも挑戦
4. **共有**: チームに展開して、品質を組織全体で向上

### さいごに

**レン**: ユイ、今日 1 日で Playwright の基本が分かったね。AI と組み合わせると、こんなに簡単なんだ。

**ユイ**: はい！重要なのは「何をテストしたいか」を考えることで、技術的な実装は AI が助けてくれるんですね。

**レン**: その通り。これからはもっと進化して、AI が自動でテストケースを考えたり、バグを見つけたりする時代が来るよ。

**ユイ**: 楽しみです！まずは今日教えてもらったことを、自分のプロジェクトで試してみます。

**レン**: がんばって。分からないことがあれば、いつでも AI に聞けばいい。**AI は 24 時間働いてくれる最高の相棒**だからね。

---

## 📚 参考リンク

- [Playwright 公式ドキュメント](https://playwright.dev/)
- [GitHub Copilot](https://github.com/features/copilot)
- [Playwright Best Practices](https://playwright.dev/docs/best-practices)

---

## 🎯 次回予告

**ユイとレン先輩の AI 駆動開発シリーズ**、次回は…

### 第 2 話:「GitHub Copilot Chat の隠れた便利機能 10 選」

**ユイ**: 先輩、Copilot Chat って基本的な質問しかしてないんですけど、もっと活用できますか?

**レン**: もちろん！実は 90% の人が知らない「隠れた便利機能」があるんだ。たとえば…

- コード全体のリファクタリング提案
- セキュリティ脆弱性の自動検出
- 「このエラーを解決して」で自動修正
- テストカバレッジ向上の提案
- コミットメッセージの自動生成

**ユイ**: え、そんなことまでできるんですか！

**レン**: うん。次回はこれを全部、実例付きで教えるよ。

👉 **次回もお楽しみに！**

---

## 💬 この記事について

**この記事が役に立ったら、ぜひシェア・フォローをお願いします！** 🎉

### 📢 AI Narrative Studio をフォロー

- **X (Twitter)**: [@ai_narrative25](https://twitter.com/ai_narrative25)
- **note**: [ai_narrative25](https://note.com/ai_narrative25)

### 🏷️ タグ

#AI駆動開発 #Playwright #テスト自動化 #GitHubCopilot #初心者向け #対話形式 #E2Eテスト

---

## ✍️ 著者について

**AI Narrative Studio** — AI と人間の対話で、技術を物語のように学ぶ。

- 7 システム × 米国株自動売買の開発者
- Playwright × MCP/Copilot による E2E テスト自動化
- 検証ループ自動化 (auto_refine_loop) の実装経験

**運営プロジェクト**: [quant_trading_system](https://github.com/r-estoer3588/quant_trading_system_0510to0906) — 開発・検証中の自動売買システム (4 long + 3 short strategies)

---

## 🌟 関連記事（近日公開）

- 【対話で学ぶ】GitHub Copilot Chat の隠れた便利機能 10 選
- 【対話で学ぶ】エラーメッセージの読み方入門
- 【対話で学ぶ】7 システム × 米国株自動売買の舞台裏
- 【対話で学ぶ】MCP × Copilot で検証ループ自動化

---

**この記事は 2025 年 10 月 13 日に公開されました。**
**最終更新: 2025 年 10 月 13 日**
````
