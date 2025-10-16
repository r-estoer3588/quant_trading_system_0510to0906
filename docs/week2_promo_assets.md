# Week2 プロモーション & 運用パッケージ

対象記事: 【対話で学ぶ】Playwright の CI 失敗を 30 分で潰す実践ガイド (Week 2 / 4)
公開予定: 2025-10-25 18:00

---

## 1. X（Twitter）投稿案

### 1-1. 50 字要約（速報向け）

CI 落ちの“型”は 4 つ。タイムアウト/セレクタ/環境差/並列競合を 30 分で片付ける実践ガイド。

### 1-2. 100 字要約（本編誘導）

Playwright の CI 失敗、毎回つらい…を 4 パターンに分解。ログ → 原因 → 再現 → 対策まで、Copilot Chat を“現場の相棒”として回す実務テンプレをまとめました。

### 1-3. 5 連スレッド案

1/5 CI で E2E が落ちる…を 4 つに分解（時間/セレクタ/環境/並列）。まずは“型”に当てはめる。

2/5 タイムアウト: 固定待機 → 条件待機。`visible`/`attached` の選定が肝。

3/5 セレクタ: `getByRole` 優先、曖昧さは `getByTestId`。

4/5 環境差: ビューポート/フォント/タイムゾーン固定。比較の閾値も“現実的”に。

5/5 並列競合: データ分離・serial 切り出し・一時テナント。↓ 本文（リンク）

（公開後 URL を 5/5 の末尾に追記）

### 1-4. 推奨ハッシュタグセット（最大 8）

#Playwright #CI #GitHubActions #GitHubCopilot #CopilotChat #テスト自動化 #開発効率化 #ログ解析

### 1-5. ポスト用画像 ALT テキスト

対話形式で学ぶ Playwright の CI 失敗対応。ユイとレン先輩の会話。青と白の背景にタイトルテキスト。

---

## 2. LinkedIn 投稿案

### 2-1. 日本語（~500 文字）

Playwright の CI 失敗を“型”で処理する実務ガイド。タイムアウト/セレクタ/環境差/並列競合の 4 パターンに分解し、ログから原因 → 再現 → 修正 → 回避まで 30 分で到達する手順と、Copilot Chat を活用した会話テンプレをまとめました。

### 2-2. 英語（~700 chars）

A practical guide to fixing flaky Playwright CI in 30 minutes. Break failures into four patterns—Timing, Selectors, Environment, Concurrency—and drive from logs to root cause, repro, fix, and prevention, aided by GitHub Copilot Chat prompts.

### 2-3. 推奨タグ（5–7）

#Playwright #ContinuousIntegration #Testing #GitHubActions #DeveloperExperience #Refactoring #Automation

---

## 3. note 貼り付けチェックリスト

- [ ] 公開直前に Week1 実 URL をシリーズ一覧へ反映
- [ ] 余計な空白・HTML コメントなし
- [ ] 章見出しがモバイルで折返し崩れない
- [ ] 画像/コードの横スクロールが必要最小限

---

## 4. KPI 計測テンプレ（Week2）

- 初速: T+6h / T+24h / T+72h / T+7d で Week1 と比較
- 主要指標: PV / 滞在 / 完読 Proxy / いいね / シェア / タグクリック
- 低調ならタイトル差替え（CI/実務/30 分の 3 要素を強調）

---

## 5. 次アクション TODO

- [ ] 公開 URL を 1-3 スレッド末尾に追記
- [ ] アイキャッチ ALT 文を note へ反映
- [ ] 公開 24h 後に改善案を Week3 企画へ反映
