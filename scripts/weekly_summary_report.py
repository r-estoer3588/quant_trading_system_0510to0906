"""Weekly Summary Report - Weekend portfolio review.

Scheduled to run every Saturday at 10:00 JST.
Provides weekly performance summary via Slack.

Usage:
    python scripts/weekly_summary_report.py [--paper]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any

from common import broker_alpaca as ba
from common.notifier import create_notifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results_csv"


def get_week_trades() -> list[dict[str, Any]]:
    """Get trades from this week's sent markers."""
    try:
        sent_path = DATA_DIR / "alpaca_sent_markers.json"
        if not sent_path.exists():
            return []

        with sent_path.open("r", encoding="utf8") as f:
            markers = json.load(f)

        # Get this week's dates
        today = datetime.now().date()
        week_start = today - timedelta(days=today.weekday())

        trades = []
        for key, value in markers.items():
            when_str = value.get("when", "")
            if when_str:
                try:
                    when = datetime.fromisoformat(when_str).date()
                    if when >= week_start:
                        symbol = key.split("_")[0]
                        trades.append(
                            {
                                "symbol": symbol,
                                "date": when.isoformat(),
                            }
                        )
                except Exception:
                    pass

        return trades
    except Exception as e:
        logger.error(f"Failed to get week trades: {e}")
        return []


def get_account_history(paper: bool = True) -> dict[str, Any]:
    """Get account equity history for the week."""
    try:
        client = ba.get_client(paper=paper)
        account = client.get_account()

        return {
            "equity": float(getattr(account, "equity", 0) or 0),
            "cash": float(getattr(account, "cash", 0) or 0),
            "last_equity": float(getattr(account, "last_equity", 0) or 0),
        }
    except Exception as e:
        logger.error(f"Failed to get account history: {e}")
        return {}


def build_weekly_report(paper: bool = True) -> str:
    """Build weekly summary report."""
    account = get_account_history(paper)
    trades = get_week_trades()

    equity = account.get("equity", 0)

    # Week date range
    today = datetime.now().date()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=4)  # Friday

    report = f"""📊 【週次サマリーレポート】

📅 **期間**: {week_start.strftime('%Y/%m/%d')} 〜 {week_end.strftime('%Y/%m/%d')}

💰 **現在のポートフォリオ**
• 口座残高: ${equity:,.2f}

📈 **今週のトレード**
• エグジット数: {len(trades)}件"""

    if trades:
        report += "\n\n🔻 **エグジット銘柄**"
        for t in trades[:10]:
            report += f"\n• {t['symbol']} ({t['date']})"
        if len(trades) > 10:
            report += f"\n• ...他{len(trades) - 10}件"

    report += "\n\n✅ 週次レポート完了"

    return report


def send_weekly_report(paper: bool = True) -> None:
    """Send weekly report via Slack."""
    report = build_weekly_report(paper)

    print(report)

    try:
        notifier = create_notifier(platform="slack", fallback=True)
        notifier.send("📊 週次サマリーレポート", report, channel=None)
        logger.info("Weekly report sent")
    except Exception as e:
        logger.error(f"Failed to send report: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", default=True)
    args = parser.parse_args()

    send_weekly_report(paper=args.paper)


if __name__ == "__main__":
    main()
