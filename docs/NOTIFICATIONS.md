# 通知設定（Discord/Slack/Teams）

以下の環境変数を `.env` に設定すると、各種通知が有効になります。

- `DISCORD_WEBHOOK_URL`（推奨）
- `SLACK_WEBHOOK_URL`
- `TEAMS_WEBHOOK_URL`

主な送信元:

- `common/notifier.py`: Discord/Slack にリッチ埋め込みで送信
- `tools/notify_signals.py`: Discord/Slack/Teams にシンプルなテキストで送信
- `scripts/tickers_loader.py`: ティッカー更新の通知を送信

補足:

- Slack が設定されている場合は Slack を優先し、なければ Discord を自動選択します。
- Teams はプレーンテキスト送信のみ対応します。
