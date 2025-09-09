# Changelog

本プロジェクトの変更履歴を SemVer に従って記録します。

## Unreleased
- 初期: 開発ツール導入（pre-commit, ruff, black, isort, mypy）
- CI 追加（lint/type/security/test）
- ドキュメント整備（AGENTS.md, .env.example）
- Slack通知が送信されない問題を修正（textフィールドを追加）
- バッチタブに当日シグナル実行を追加
- Alpaca ステータスダッシュボードを追加
- 売買通知をBUY/SELLで集約し、数量と金額を含めて表示
- 既存キャッシュが当日更新済みでも recent キャッシュを生成するよう修正
- シグナル計算で必要日数分の履歴をUIで読み込み `symbol_data` として渡すよう調整
- 当日シグナル画面で保有ポジションと利益保護判定を表示し、ボタン操作で更新可能に
