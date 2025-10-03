"""Slack Notifier デバッグ最小スクリプト

目的:
  - Slack 通知が届かない/沈黙する原因を即座に切り分ける
  - 必須環境変数 (SLACK_BOT_TOKEN / SLACK_CHANNEL*) の有無と簡易送信結果を可視化
  - _slack_send_text に追加した debug reasons ([SLACK_DEBUG]) を確実に表示

使い方 (PowerShell / CMD 共通):
  python scripts/debug_slack_notifier.py --text "テスト" --channel "#your-channel"

追加オプション:
  --rich        : RichSlackNotifier 経由 (Block Kit) で送信試行
  --channel CH  : 明示チャンネル (未指定なら 環境変数 優先探索)
  --text TXT    : 送信本文

前提:
  - slack_sdk がインストールされていること
  - SLACK_BOT_TOKEN が有効 (xoxb- で始まる) であること
  - Bot が指定チャンネルに参加済み (未参加なら /invite)

表示される可能性のある代表的 debug reason:
  - Slack SDK 未インポート / トークン未設定 / チャンネル未設定 / API 例外(channel_not_found, not_in_channel, invalid_auth など)

このスクリプトは本番ロジックを変更しません。
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Optional


def _print_header():
    print("================ Slack Debug Notifier =================")
    print("Python:", sys.version.split()[0])
    print("Working Dir:", os.getcwd())
    print("File:", __file__)
    print("--------------------------------------------------------")


def _summarize_env():
    token = os.getenv("SLACK_BOT_TOKEN")
    channel_fallbacks = [
        ("SLACK_CHANNEL", os.getenv("SLACK_CHANNEL")),
        ("SLACK_CHANNEL_ID", os.getenv("SLACK_CHANNEL_ID")),
        ("SLACK_CHANNEL_SIGNALS", os.getenv("SLACK_CHANNEL_SIGNALS")),
        ("SLACK_CHANNEL_EQUITY", os.getenv("SLACK_CHANNEL_EQUITY")),
        ("SLACK_CHANNEL_LOGS", os.getenv("SLACK_CHANNEL_LOGS")),
    ]
    print("[ENV] SLACK_DEBUG_VERBOSE=", os.getenv("SLACK_DEBUG_VERBOSE"))
    print("[ENV] NOTIFIER_LOG_LEVEL =", os.getenv("NOTIFIER_LOG_LEVEL"))
    if token:
        masked = token[:10] + "..." + token[-5:]
        print("[ENV] SLACK_BOT_TOKEN   = PRESENT (", masked, ")")
    else:
        print("[ENV] SLACK_BOT_TOKEN   = MISSING")
    for k, v in channel_fallbacks:
        if v:
            print(f"[ENV] {k:20s} = {v}")
    print("--------------------------------------------------------")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Slack notifier debug runner")
    parser.add_argument("--text", default="デバッグ送信テスト", help="送信本文")
    parser.add_argument("--channel", default=None, help="明示チャンネル (#name or Cxxxx)")
    parser.add_argument("--rich", action="store_true", help="RichSlackNotifier を使用")
    args = parser.parse_args(argv)

    # デバッグ環境変数を強制 (既に設定されていれば尊重)
    os.environ.setdefault("SLACK_DEBUG_VERBOSE", "1")
    os.environ.setdefault("NOTIFIER_LOG_LEVEL", "DEBUG")

    _print_header()
    _summarize_env()

    try:
        from common.notifier import RichSlackNotifier, SimpleSlackNotifier
    except Exception as e:  # pragma: no cover - 診断専用
        print("[FATAL] notifier import failed:", e)
        traceback.print_exc()
        return 2

    notifier_cls = RichSlackNotifier if args.rich else SimpleSlackNotifier
    print(f"[INFO] Using notifier class: {notifier_cls.__name__}")

    try:
        notifier = notifier_cls()
    except Exception as e:  # pragma: no cover - 初期化失敗診断
        print("[FATAL] Notifier init failed:", e)
        traceback.print_exc()
        return 3

    print("[ACTION] Sending test message...")
    try:
        if args.channel:
            # 現行公開 API は channel を send_signals 等で個別指定できるが、汎用メッセージは
            # デフォルトチャンネル解決に従うため env を上書きして対応する。
            os.environ["SLACK_CHANNEL"] = args.channel
        # SimpleSlackNotifier/RichSlackNotifier 共通で存在する send(title, message)
        notifier.send("DEBUG Slack Test", args.text)
    except Exception as e:  # pragma: no cover - 送信例外もそのまま表示
        print("[EXCEPTION] send raised:", e)
        traceback.print_exc()
        return 4

    print("[DONE] 呼び出し完了。 [SLACK_DEBUG] 行に理由・結果が出ているか確認してください。")
    print("========================================================")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
