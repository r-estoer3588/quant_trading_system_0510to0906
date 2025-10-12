# AI ダブルチェック運用フロー

このリポジトリでは、コードの実装・修正後に「別の AI モデル」によるダブルチェックを行います。目的は、ヒューマン＋ AI の見逃しを減らし、品質/安全性/一貫性を高めることです。

## 対象

- すべての機能追加・不具合修正・仕様変更
- ドキュメントの重要更新（運用/アーキテクチャ/危険設定など）

## 手順（毎回実施）

1. 開発者（または本 AI）が変更を実装し、最小テストを実行
   - Lint/Format/Tests をローカルで PASS にする
   - 影響する機能に最小限のユニットテストを追加
2. ダブルチェック要求を記録（Pull Request またはコミットコメント）
   - 変更概要、懸念点、想定影響範囲、再現手順を簡潔に記述
3. 別 AI モデルでレビュー（自動/半自動）
   - 実装に使用した AI モデル“以外”を選ぶ（モデル分離ルール）
   - 変更差分の静的レビュー（スタイル・型・セキュリティ）
   - 実行検証（対象テストのみ/mini パイプライン）
   - ガードレール遵守の検証
     - CacheManager 経由 I/O、System7/SPY 固定、CLI 互換、--test-mode/--skip-external 互換、環境変数アクセサ使用（get_env_config）
4. フィードバック反映
   - 指摘は可能な限り即時反映し、再度 Lint/Tests を PASS
5. 完了記録
   - PR に「AI レビュー完了」ラベルとチェックリスト結果を残す

## チェックリスト（要全項目 OK）

- [ ] ビルド・Lint・テストは PASS
- [ ] 仕様の不変条件を満たす（System7=SPY 固定、Two-Phase 整合、DEFAULT_ALLOCATIONS 維持）
- [ ] キャッシュ I/O は CacheManager 経由のみ
- [ ] 外部ネットワークをテスト経路に追加していない
- [ ] today_signals → allocation → trade_management で Entry/Stop/ATR/株数が一貫
- [ ] pandas FutureWarning を出していない（空結合/全 NA 対策済）
- [ ] env は get_env_config() 経由で参照し os.environ.get を直接使用していない
- [ ] 公開 API/CLI フラグの互換性維持

## 使用するレビュー用 AI モデル（いずれか）

以下のいずれかを使用し、実装・修正に用いたモデルとは異なるモデルを選択します。

- Claude sonnet 4.5
- gpt-5
- gpt-5-codex

記録ルール（PR/コミットに明記）

- Implementer Model: 実装で使用したモデル名
- Reviewer Model: ダブルチェックに使用したモデル名（Implementer と異なること）
- 差分要約 / 指摘事項 / 対応可否

## 実行方法（運用）

- VS Code タスク
  - 「Lint & Format」「Quick Test Run」「Run All Systems Today (mini)」の順で実行
- PR テンプレート（推奨）
  - 変更概要、リスク、ロールバック、テスト結果、AI レビュー結果を記載
- 失敗時
  - 指摘事項を修正し、再度ダブルチェックを要求

## 備考

- 別 AI モデルの運用は将来的に CI 統合を予定（例：PR イベントで自動起動）
- 大規模変更は段階的に PR 分割し、各段階でダブルチェックを行う
