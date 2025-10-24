"""
シグナル生成結果の通知スクリプト

Slack に日次シグナル生成の結果を通知。

使い方:
    python scripts/notify_results.py --signals 25 --log logs/auto_run.log
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys


def send_notification(signal_count: int, log_file: Path):
    """通知を送信"""
    # Slack 通知（common/notification.py を使用）
    try:
        from common.notification import send_slack_message
    except ImportError:
        print("⚠️  common.notification モジュールが見つかりません")
        print("   Slack 通知をスキップします")
        return False

    # ログファイルの最終行を取得
    log_tail = ""
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            log_tail = "".join(lines[-10:])  # 最後の10行

    # メッセージ作成
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = f"""
📊 *Daily Signal Generation Complete*

**Date:** {timestamp}
**Signals Generated:** {signal_count}
**Status:** ✅ Success

**Log File:** `{log_file.name}`

**Recent Log:**
```
{log_tail}
```
    """.strip()

    try:
        # Slack チャンネルに送信
        send_slack_message(message, channel="#trading-signals")
        print("✅ Slack 通知を送信しました")
        return True
    except Exception as e:
        print(f"❌ Slack 通知の送信に失敗: {e}")
        return False


def send_email_notification(signal_count: int, log_file: Path):
    """メール通知（オプション）"""
    # 将来の拡張用
    pass


def main():
    parser = argparse.ArgumentParser(description="シグナル生成結果の通知")
    parser.add_argument(
        "--signals", type=int, required=True, help="生成されたシグナル数"
    )
    parser.add_argument("--log", type=Path, required=True, help="ログファイルのパス")
    parser.add_argument("--email", action="store_true", help="メール通知も送信")
    args = parser.parse_args()

    print("📤 通知を送信中...")
    print(f"   Signals: {args.signals}")
    print(f"   Log: {args.log}")

    # Slack 通知
    slack_success = send_notification(args.signals, args.log)

    # メール通知（オプション）
    email_success = True
    if args.email:
        email_success = send_email_notification(args.signals, args.log)

    if slack_success and email_success:
        print("✅ 全ての通知を送信しました")
        return 0
    else:
        print("⚠️  一部の通知が失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
