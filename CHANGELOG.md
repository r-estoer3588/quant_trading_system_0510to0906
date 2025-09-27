# Changelog

本プロジェクトの変更履歴を SemVer に従って記録します。

## Unreleased
- 初期: 開発ツール導入（pre-commit, ruff, black, isort, mypy）
- CI 追加（lint/type/security/test）
- ドキュメント整備（AGENTS.md, .env.example）
- Slack通知が送信されない問題を修正（textフィールドを追加）
- バッチタブに当日シグナル実行を追加
- System1〜7 指標計算を差分更新し Feather に累積キャッシュ
- `app_today_signals.py` を削除（機能をバッチタブに統合）
- Alpaca ステータスダッシュボードを追加
- 売買通知をBUY/SELLで集約し、数量と金額を含めて表示
- 既存キャッシュが当日更新済みでも recent キャッシュを生成するよう修正
- シグナル計算で必要日数分の履歴をUIで読み込み `symbol_data` として渡すよう調整
- シグナル通知に推奨銘柄の日足チャート画像を添付
- `run_all_systems_today.py` の文字列連結を改善
- Today Signals に保有ポジションと利益保護判定を追加
- `load_price` で `cache_profile="rolling/full"` 指定時に base キャッシュを自動的に挟み、フォールバック順を rolling→base→full に統一
- **データストレージ最適化**: CSV+Feather デュアルフォーマット対応（6,200+銘柄）
- **重複列削除**: 冗長データクリーンアップで40%の列削減を実現
- **CacheManager拡張**: Feather優先読み取り、CSV自動フォールバック機能
- System5/6 の利食い・時間退出ロジックを仕様どおりに修正しテストを追加
