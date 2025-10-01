# Changelog

本プロジェクトの変更履歴を SemVer に従って記録します。

## Unreleased

- 初期: 開発ツール導入（pre-commit, ruff, black, isort, mypy）
- CI 追加（lint/type/security/test）
- ドキュメント整備（AGENTS.md, .env.example）
- Slack 通知が送信されない問題を修正（text フィールドを追加）
- バッチタブに当日シグナル実行を追加
- System1〜7 指標計算を差分更新し Feather に累積キャッシュ
- `app_today_signals.py` を削除（機能をバッチタブに統合）
- Alpaca ステータスダッシュボードを追加
- 売買通知を BUY/SELL で集約し、数量と金額を含めて表示
- 既存キャッシュが当日更新済みでも recent キャッシュを生成するよう修正
- シグナル計算で必要日数分の履歴を UI で読み込み `symbol_data` として渡すよう調整
- シグナル通知に推奨銘柄の日足チャート画像を添付
- `run_all_systems_today.py` の文字列連結を改善
- Today Signals に保有ポジションと利益保護判定を追加
- `load_price` で `cache_profile="rolling/full"` 指定時に base キャッシュを自動的に挟み、フォールバック順を rolling→base→full に統一
- **データストレージ最適化**: CSV+Feather デュアルフォーマット対応（6,200+銘柄）
- **重複列削除**: 冗長データクリーンアップで 40%の列削減を実現
- **CacheManager 拡張**: Feather 優先読み取り、CSV 自動フォールバック機能
- System5/6 の利食い・時間退出ロジックを仕様どおりに修正しテストを追加
- **テスト高速化機能**: `run_all_systems_today.py` に高速テストオプション追加
  - `--test-mode mini/quick/sample`: 銘柄数制限（10/50/100 銘柄）
  - `--skip-external`: 外部 API 呼び出しをスキップして高速化
  - 実行時間を分単位から 2 秒に短縮（mini モード）
  - 4 つのテストシナリオ対応（基本/並列/CSV/統合テスト）
- **公開 API 境界保護**: `tests/test_public_api_exports.py` 追加（存在/漏洩/新規 callable 監視 + Docstring 情報）
- **Warnings ポリシー策定**: `docs/technical/warnings_policy.md` 追加し分類・段階的削減方針を明文化
- **CI 強化**: warnings 収集 + JSONL アーティファクト化、mypy エラー件数サマリ出力、`collect_warnings.py` 統合
- **実 Alpaca テスト安定化**: `test_real_alpaca.py` を `RUN_REAL_ALPACA_TEST=1` 条件付き skip 化
- **mypy ロードマップ**: `docs/technical/mypy_roadmap.md` で段階導入計画を提示（P0〜P5）
- **Warnings 集計スクリプト**: `tools/collect_warnings.py` 追加
- **UI コンポーネント防御的改善**: 欠損列ガード / matplotlib Agg / heatmap 高速モード / download key 一意化
- **不要機能撤去**: 旧 save_prepared_data_cache ロジック削除＆依存テスト整理
