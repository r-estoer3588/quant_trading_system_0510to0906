# 環境変数一覧（運用向け）

このページは本リポジトリで利用される主な環境変数を一覧化し、既定値と用途を簡潔にまとめたものです。詳細は `config/settings.py` および各モジュールの実装を参照してください。

> 備考: 値の真偽判定は基本的に "1" / "true" / "yes" / "on" を真と見なします。

## グローバル/共通

- COMPACT_TODAY_LOGS: 既定 "0"。当日シグナル系ログを簡略化。
- ENABLE_PROGRESS_EVENTS: 既定 "false"。ステージ進捗イベントを有効化。
- STRUCTURED_UI_LOGS: 既定 "0"。UI 向けに構造化ログ(JSON 文字列)を出力。
- STRUCTURED_LOG_NDJSON: 既定 "0"。NDJSON 形式で `logs/` へ構造化ログ出力。
- SHOW_INDICATOR_LOGS: 既定 "0"。インジケータ計算進捗ログの出力を許可。
- TODAY_SIGNALS_LOG_MODE: 既定 "single" or env。`dated` で日付別ファイルに出力。
- DATA_CACHE_DIR: データキャッシュのルート。既定 `data_cache`。
- RESULTS_DIR: 出力ルート。既定 `results_csv`。
- LOGS_DIR: ログ出力先ルート。既定 `logs`。

## キャッシュ/IO

- ROLLING_ISSUES_VERBOSE_HEAD: 既定 "5"。rolling キャッシュ警告の先頭行数。
- CACHE_HEALTH_SILENT: 既定 "0"。キャッシュ健康診断の CLI 通知を抑制。
- BASIC_DATA_PARALLEL: 既定 自動。"1" で強制並列、"0" で直列。
- BASIC_DATA_PARALLEL_THRESHOLD: 既定 "200"。並列化する銘柄数閾値。
- BASIC_DATA_MAX_WORKERS: 明示ワーカー数。未設定時は CPU/設定から推定。

## Strategy/System 関連

- MIN_ATR_RATIO_FOR_TEST: System3 の ATR 比率しきい値テスト上書き。
- MIN_DROP3D_FOR_TEST: System3 の drop3d しきい値テスト上書き。
- VALIDATE_SETUP_PREDICATE: 既定 "0"。predicate と setup 列の一致検証を有効化。

## 通知/ダッシュボード

- SLACK*BOT_TOKEN, SLACK_CHANNEL*\*, DISCORD_WEBHOOK_URL: 通知連携。
- NOTIFY_USE_RICH: 既定 "0"。通知をリッチカード形式で送信。

## ベンチマーク/パフォーマンス

- ENABLE_STEP_TIMINGS: 既定 "0"。段階ごとの処理時間を DEBUG で出力。

---

更新ガイド:

- 新しい環境変数を導入する場合はここへ追記し、実装部に既定値の根拠コメントを残してください。
