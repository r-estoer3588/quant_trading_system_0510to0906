<!-- docs/branding/content_calendar.md -->

# 30 日運用プラン（週 3 本 × 4 週）

方針

- 各週：3 本の柱を均等配分（AI 駆動開発 / 米国株自動売買 / Playwright×E2E 検証）
- 各回：短くても"学び 1 つ＋添付テンプレ 1 つ"を必ず入れる

Week 1

- AI 駆動: MCP/Copilot 連携で検証ループを自動化（失敗 → 差分 → 提案の流れ）
- 自動売買: Two-Phase 設計（Filter→Setup→Rank→Allocate）の検証観点
- E2E 検証: Playwright の最小落ちない構成（fixtures・安定化フラグ・動画/スクショ）

Week 2

- AI 駆動: プロンプトの再利用設計（変数化・条件分岐・安全弁）
- 自動売買: キャッシュ IO の安全化（Feather 優先＋ CSV fallback の理由）
- E2E 検証: 失敗時に読むのはログではなく"差分"（スクショ ×DOM 変化）

Week 3

- AI 駆動: Copilot を仕様化に使う（疑似テストケース生成 → 削る）
- 自動売買: 日次パイプラインの"テストモード mini"思想（データ依存を切る）
- E2E 検証: Playwright の安定待ち（locator / expect / retry）。典型アンチパターン集

Week 4

- AI 駆動: 画像自動生成をワークフロー化（ヘッダー/アイコン自動再出力）
- 自動売買: 失敗学まとめ（不確実性と検証ループ）
- E2E 検証: CI で"速いが壊れない"テストの本数を作る（1 分以下の核）
