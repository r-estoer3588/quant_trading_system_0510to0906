<!--
  Article Meta Block (編集時テンプレ)
  公開時に必要な要素: 公開日(published), 最終更新(updated), タグ(tags), シリーズ(series)
  更新時運用ルール:
    1. 軽微な誤字修正は updated をそのまま(週次まとめ時に一括更新)
    2. 章差替え/内容追加は updated を即日へ
  シリーズ識別: series_slug を共通利用 (Week1-4 で統一)
-->

> 📅 公開日: 2025-10-25 18:00 / 最終更新: 2025-10-15
> 🏷️ タグ: Playwright, CI, テスト自動化, GitHub Actions, GitHub Copilot, Copilot Chat, 開発効率化, ログ解析
> 🗂️ シリーズ: 対話で学ぶ Copilot 活用術 (Week 2 / 4)

# 【対話で学ぶ】Playwright の CI 失敗を 30 分で潰す実践ガイド

**副題**: タイムアウト・セレクタ・環境差・並列競合を「会話」で片付ける

---

## 導入

ユイ: 先輩、CI で E2E テストがよく落ちて…毎回ログを読むのに 1 時間かかります。

レン: よくある悩みだね。今日は失敗を 4 パターンに分けて、Copilot Chat を“現場の相棒”としてどう使うかを実践で整理しよう。目標は「30 分で原因特定 → 対策」だよ。

---

## 登場人物と今日のゴール

- ユイ: Playwright 初心者（導入 3 ヶ月）
- レン: QA/Automation リード（実務 5 年）

> 今日のゴール
>
> 1. 失敗を 4 軸（時間/セレクタ/環境/並列）で即分類
> 2. 再現・修正・回避の“次アクション”を 30 分以内に決定

---

## 失敗パターン 1: タイムアウト地獄からの脱出

### 症状

- TimeoutError: waiting for selector ...
- 遅い環境のみ再現 / まれに通る

### 会話（抜粋）

ユイ: `waitForTimeout(5000)` を足しても落ちます…。

レン: 固定待機は最終手段。条件待ちに置き換えよう。API 後の表示なら `state: "visible"` をまず試して。

### プロンプト例（Copilot へ）

```
次の Playwright コードで時々 TimeoutError になります。高速/低速環境で安定する待機戦略に書き換えてください。理由も 3 行で。

（※ 問題のテストコードを貼り付け）
```

### 修正例

```ts
// ❌ 固定待機
await page.waitForTimeout(5000);
await page.locator("#result").click();

// ⭕ 条件待ち + 自動待機の活用
await page.waitForSelector("#result", { state: "visible" });
await page.getByRole("button", { name: "結果を表示" }).click();
```

### 補足

- `attached` は DOM だけ、`visible` は描画まで。アニメありは visible 推奨。
- 画面遷移は `page.waitForURL()` / `locator.isVisible()` 併用が堅い。

---

## 失敗パターン 2: セレクタが突然壊れる

### 症状

- class/innerText 依存でデザイン変更や i18n で崩れる

### 会話（抜粋）

ユイ: `.btn-primary` にしてました…。

レン: 実運用は `getByRole` 優先。曖昧さが残るときだけ `getByTestId` を使う。アクセスビリティ改善にも効くよ。

### プロンプト例

```
次のセレクタを role/testid 中心で安定化してください。i18n でも壊れにくい戦略に。

（※ 対象コードを貼り付け）
```

### 修正例

```ts
// ❌ テキスト/クラス依存
await page.locator("text=送信").click();
await page.locator(".btn-primary").click();

// ⭕ role/testid
await page.getByRole("button", { name: /送信|Submit/ }).click();
await page.getByTestId("submit-btn").click();
```

### 補足

- name は正規表現で多言語吸収。ARIA label を整備すると UI/テスト双方が安定。

---

## 失敗パターン 3: ローカル OK / CI NG の環境差

### 症状

- スクショ/レイアウト差・フォント/解像度差で比較失敗

### 会話（抜粋）

ユイ: ローカルは通るのに Ubuntu CI でだけ落ちます…。

レン: 環境起因の匂い。ビューポート/デバイス/フォントを固定しよう。閾値やマージンを現実的に緩めるのも手。

### プロンプト例

```
CI(ubuntu) とローカル(macOS)でスクショ比較が不安定です。ビューポート/フォント/デバイス設定を最小セットで固定した playwright.config.ts を提案してください。
```

### 設定例（最小）

```ts
import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  use: {
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 1,
    locale: "ja-JP",
    timezoneId: "Asia/Tokyo",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
```

### 補足

- 画像比較は `maxDiffPixelRatio` などで“許容ノイズ”を定義。

---

## 失敗パターン 4: 並列実行での競合

### 症状

- DB ロック/テスト間干渉/共有リソース競合

### 会話（抜粋）

ユイ: 並列だとだけ DB 関連が落ちます。

レン: データ分離が必要。テストごとに独立データ/一時スキーマ/ユーザー名に UUID 接尾辞など。`serial` プロジェクトへの切り出しも検討。

### プロンプト例

```
並列で落ちる E2E テストを、データ分離と直列化で安定化する設計ガイドを 5 箇条でください。
```

### 対策例

- 共有データの書き換え禁止 → 各テストで作成/破棄
- CI 並列数を段階的に引き上げ、閾値を把握
- 競合しやすいシナリオは `serial` プロジェクトで分離
- 一時ユーザー/テナント名にランダムサフィックス
- 外部 API はフェイク/モックを優先

---

## ログを“要点化”するテンプレ（共通）

### プロンプト（コピペ可）

```
以下の CI ログから、
1) 根本原因の仮説
2) 再現ステップ
3) 優先度（高/中/低）と先に潰すべき理由
を 10 行以内で整理してください。必要なら修正案のコードも添えてください。

（※ ログ全文を貼り付け）
```

---

## 現場で使えるチェックリスト（10 項目）

1. 固定待機を条件待機へ置換したか（`visible/attached` の選定理由は）
2. セレクタは role/testid 優先か（i18n/デザイン変更で壊れないか）
3. ビューポート/フォント/タイムゾーンは固定したか
4. 画像比較のしきい値は“現実的”か（誤検知を抑制）
5. 並列で競合するテストを切り出したか（serial/タグ）
6. データ分離（UUID 接尾辞/一時スキーマ/テナント分離）
7. 外部 API はモック化/リトライ設定を適用したか
8. ログ収集に「原因/再現/優先度」テンプレを使ったか
9. CI の失敗を 環境/時間/データ の 3 軸で分類したか
10. 再発防止を config/practice に恒久反映したか

---

## まとめ: 30 分で“次に進む”判断を

- 4 パターンの観点で分類 → 再現・修正・回避を素早く決定
- Copilot Chat は「読み解き」と「設計の比較検討」を加速
- 仕組みの反映（config/テンプレ/命名規則）で再発を止める

---

## 次回予告 📢

次回は「落ちないテスト設計の実践」。テストデータ設計（境界値/同値分割）、失敗の再現性を上げる戦略、保守しやすい `test.each` パターンを、会話で一気に形にします。

---

### シリーズ記事一覧

- **Week 1**: [Copilot Chat の隠れた便利機能 10 選](https://note.com/ai_narrative25/n/nd305530c5064)
- **Week 2**: Playwright の CI 失敗を 30 分で潰す実践ガイド ← 本記事
- **Week 3**: 落ちないテスト設計の実践（近日公開）
- **Week 4**: AI × E2E の運用設計（近日公開）

---

### 🏷️ Tags (note 用表示)

`#Playwright` `#CI` `#GitHubActions` `#GitHubCopilot` `#CopilotChat` `#テスト自動化` `#開発効率化` `#ログ解析`
