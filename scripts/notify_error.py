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


def send_error_notification(error_message: str, log_file: Path):
    """エラー通知を送信"""
    try:
        from common.notification import send_slack_message
    except ImportError:
        print("⚠️  common.notification モジュールが見つかりません")
        print("   Slack 通知をスキップします")
        return False

    # ログファイルの最終行を取得
    log_tail = ""
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            log_tail = "".join(lines[-20:])  # 最後の20行

    # メッセージ作成
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = f"""
🚨 *Daily Signal Generation FAILED*

**Date:** {timestamp}
**Error:** {error_message}
**Status:** ❌ Failed

**Log File:** `{log_file.name}`

**Error Log:**
```
{log_tail}
```

**Action Required:** Please check the log file and investigate the issue.
    """.strip()

    try:
        # Slack の #trading-errors チャンネルに送信
        send_slack_message(message, channel="#trading-errors")
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

    success = send_error_notification(args.error, args.log)

    # エラー通知の失敗は致命的ではないため、常に0を返す
    return 0


if __name__ == "__main__":
    sys.exit(main())
