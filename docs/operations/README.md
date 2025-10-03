# 運用ガイド

システムの運用に関するドキュメント集です。

## 🔄 自動実行設定

### [スケジュール設定（クイックスタート）](./schedule_quick_start.md)

- Windows Task Scheduler での自動実行設定
- 平日朝の定期実行
- テスト実行とトラブルシューティング

### [Windows 自動実行詳細](./run_auto_rule_windows.md)

- PowerShell スクリプト設定
- 環境変数の管理
- エラーハンドリング

## 📢 通知設定

### [通知システム](./NOTIFICATIONS.md)

- Slack Webhook 連携
- Discord 通知
- エラー通知とアラート設定

## 📊 監視・メトリクス

### [UI メトリクス](./today_signals_ui_metrics.md)

- ダッシュボード機能
- パフォーマンス監視
- データ品質チェック

### 日次メトリクス

- **出力先**: `results_csv/daily_metrics.csv`
- **生成**: `scripts/run_all_systems_today.py` 実行時
- **用途**: フィルター通過数と候補数の日次監視

## 🔗 関連リンク

- [システム概要](../systems/) - 各システム仕様
- [技術文書](../technical/) - 内部実装詳細
- [今日のシグナル処理](../today_signal_scan/) - 実行フロー
