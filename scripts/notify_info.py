"""
情報通知スクリプト（Slack）

使い方:
    python scripts/notify_info.py --title "市場休場" --message "休場のためスキップ"
"""

from __future__ import annotations

import argparse
import sys


def send_info(title: str, message: str, channel: str = "#trading-signals") -> bool:
    try:
        from common.notifier import create_notifier
    except Exception as e:
        print(f"⚠️ common.notifier が見つからないため Slack 通知をスキップします: {e}")
        return False

    try:
        notifier = create_notifier()
        if notifier is None:
            print("⚠️ Notifier の初期化に失敗しました")
            return False

        # Slack チャンネルを環境変数から取得（デフォルトは SLACK_CHANNEL_LOGS）
        import os

        target_channel = os.getenv("SLACK_CHANNEL_LOGS", channel)

        # titleとmessageをNotifier.send()に渡す
        notifier.send(title=f"ℹ️ {title}", message=message, channel=target_channel)
        print(f"✅ Slack 情報通知を送信しました (channel: {target_channel})")
        return True
    except Exception as e:  # pragma: no cover - runtime only
        print(f"❌ Slack 通知に失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--title", required=True)
    p.add_argument("--message", required=True)
    p.add_argument("--channel", default="#trading-signals")
    args = p.parse_args()

    ok = send_info(args.title, args.message, channel=args.channel)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
