"""
エラー通知スクリプト

エラー発生時に Slack に通知。

使い方:
    python scripts/notify_error.py --error "Database connection failed" --log logs/auto_run.log
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

from common.notifier import Notifier


def send_error_notification(error_message: str, log_file: Path) -> bool:
    """エラー通知を送信（Slack/Discord 自動判定）。"""
    notifier = Notifier(platform="auto")

    # ログファイルの最終行を取得
    log_tail = ""
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            log_tail = "".join(lines[-20:])  # 最後の20行

    # メッセージ作成
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    title = "🚨 Daily Signal Generation FAILED"
    message = (
        f"Date: {timestamp}\n"
        f"Error: {error_message}\n"
        f"Status: ❌ Failed\n\n"
        f"Log File: {log_file.name}\n\n"
        "Error Log:\n" + ("```\n" + log_tail + "\n```\n" if log_tail else "(no recent log lines)\n")
    )

    try:
        # 既定：logs チャンネル相当へ。チャンネル固定したい場合は .env で SLACK_CHANNEL_LOGS などを設定。
        notifier.send(title=title, message=message, channel="#trading-errors")
        print("✅ エラー通知を送信しました")
        return True
    except Exception as e:
        print(f"❌ エラー通知の送信に失敗: {e}")
        print("   Slack への通知が失敗しましたが、処理は続行します")
        return False


def main():
    parser = argparse.ArgumentParser(description="エラー通知")
    parser.add_argument("--error", type=str, required=True, help="エラーメッセージ")
    parser.add_argument("--log", type=Path, required=True, help="ログファイルのパス")
    args = parser.parse_args()

    print("🚨 エラー通知を送信中...")
    print(f"   Error: {args.error}")
    print(f"   Log: {args.log}")

    # 送信結果はログで十分なため、戻り値は評価しない
    send_error_notification(args.error, args.log)

    # エラー通知の失敗は致命的ではないため、常に0を返す
    return 0


if __name__ == "__main__":
    sys.exit(main())
