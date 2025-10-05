#!/usr/bin/env python
"""Alpaca Paper 口座をリセットする簡易 CLI ツール。

使い方 (PowerShell 例):
  python scripts/reset_paper_account.py --confirm          # デフォルト初期残高で実リセット
  python scripts/reset_paper_account.py --equity 150000 --confirm
  python scripts/reset_paper_account.py --equity 120000 --dry-run  # 送信せず検証

環境変数 / .env:
  APCA_API_KEY_ID, APCA_API_SECRET_KEY を参照。

注意:
  - Paper 環境のみ。Live キーでは 403/404 になります。
  - API の仕様変更で失敗する可能性があります (best-effort)。
"""

from __future__ import annotations

import argparse
import json
import sys

from common.broker_alpaca import reset_paper_account


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reset Alpaca Paper account")
    p.add_argument("--equity", type=float, default=None, help="希望初期残高 (省略可)")
    p.add_argument("--dry-run", action="store_true", help="送信せず検証のみ")
    p.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="リセット API エンドポイントを明示指定 (デフォルト/環境変数上書き用)",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="実行確認フラグ (指定がない場合は安全のため中止)",
    )
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP タイムアウト秒")
    return p.parse_args()


def main() -> int:
    ns = parse_args()
    if not ns.confirm and not ns.dry_run:
        print("--confirm なしのため中止 (--dry-run か --confirm を指定してください)")
        return 2

    result = reset_paper_account(
        desired_equity=ns.equity,
        dry_run=ns.dry_run,
        timeout=ns.timeout,
        endpoint=ns.endpoint,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not result.get("ok"):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - 手動実行想定
    sys.exit(main())
