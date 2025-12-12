"""Daily Summary Report - Send portfolio status via Slack.

Scheduled to run daily at 17:00 JST to provide:
- Portfolio balance and PnL
- Position breakdown
- Today's exits summary
- System performance

Usage:
    python scripts/daily_summary_report.py [--paper]
"""

from __future__ import annotations

import argparse
from datetime import datetime
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


def get_account_info(paper: bool = True) -> dict[str, Any]:
    """Get account information from Alpaca."""
    try:
        client = ba.get_client(paper=paper)
        account = client.get_account()

        return {
            "equity": float(getattr(account, "equity", 0) or 0),
            "cash": float(getattr(account, "cash", 0) or 0),
            "buying_power": float(getattr(account, "buying_power", 0) or 0),
            "portfolio_value": float(getattr(account, "portfolio_value", 0) or 0),
            "last_equity": float(getattr(account, "last_equity", 0) or 0),
        }
    except Exception as e:
        logger.error(f"Failed to get account info: {e}")
        return {}


def get_positions_summary(paper: bool = True) -> dict[str, Any]:
    """Get positions summary from Alpaca."""
    try:
        client = ba.get_client(paper=paper)
        positions = client.get_all_positions()

        long_count = 0
        short_count = 0
        total_unrealized_pnl = 0.0
        total_market_value = 0.0

        positions_list = []

        for p in positions:
            try:
                side = str(getattr(p, "side", "")).lower()
                unrealized_pnl = float(getattr(p, "unrealized_pl", 0) or 0)
                market_value = float(getattr(p, "market_value", 0) or 0)

                if "long" in side:
                    long_count += 1
                else:
                    short_count += 1

                total_unrealized_pnl += unrealized_pnl
                total_market_value += abs(market_value)

                positions_list.append(
                    {
                        "symbol": getattr(p, "symbol", ""),
                        "side": side,
                        "qty": int(getattr(p, "qty", 0) or 0),
                        "unrealized_pnl": unrealized_pnl,
                        "unrealized_pnl_pct": float(
                            getattr(p, "unrealized_plpc", 0) or 0
                        )
                        * 100,
                    }
                )
            except Exception:
                continue

        # Sort by PnL%
        positions_list.sort(key=lambda x: x["unrealized_pnl_pct"], reverse=True)

        return {
            "total_positions": len(positions),
            "long_count": long_count,
            "short_count": short_count,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_market_value": total_market_value,
            "top_gainers": positions_list[:3],
            "top_losers": positions_list[-3:] if len(positions_list) >= 3 else [],
        }
    except Exception as e:
        logger.error(f"Failed to get positions summary: {e}")
        return {}


def get_todays_exits() -> list[dict[str, Any]]:
    """Get today's exit records from sent markers."""
    try:
        sent_path = DATA_DIR / "alpaca_sent_markers.json"
        if not sent_path.exists():
            return []

        with sent_path.open("r", encoding="utf8") as f:
            markers = json.load(f)

        today = datetime.now().date().isoformat()
        todays = []

        for key, value in markers.items():
            if today in key:
                symbol = key.split("_")[0]
                todays.append(
                    {
                        "symbol": symbol,
                        "when": value.get("when", ""),
                    }
                )

        return todays
    except Exception as e:
        logger.error(f"Failed to get today's exits: {e}")
        return []


def build_report(paper: bool = True) -> str:
    """Build the daily summary report."""
    account = get_account_info(paper)
    positions = get_positions_summary(paper)
    exits = get_todays_exits()

    # Calculate daily PnL
    equity = account.get("equity", 0)
    last_equity = account.get("last_equity", 0)
    daily_pnl = equity - last_equity if last_equity else 0
    daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity else 0

    # Build report
    report = f"""📊 【日次サマリーレポート】{datetime.now().strftime('%Y-%m-%d')}

💰 **ポートフォリオ状況**
• 口座残高: ${equity:,.2f}
• 本日のPnL: {'+' if daily_pnl >= 0 else ''}${daily_pnl:,.2f} ({'+' if daily_pnl_pct >= 0 else ''}{daily_pnl_pct:.2f}%)
• 未決済PnL: {'+' if positions.get('total_unrealized_pnl', 0) >= 0 else ''}${positions.get('total_unrealized_pnl', 0):,.2f}
• 購買余力: ${account.get('buying_power', 0):,.2f}

📈 **ポジション内訳（{positions.get('total_positions', 0)}銘柄）**
• ロング: {positions.get('long_count', 0)}銘柄
• ショート: {positions.get('short_count', 0)}銘柄"""

    # Top gainers
    if positions.get("top_gainers"):
        report += "\n\n🏆 **本日のTop Gainers**"
        for p in positions["top_gainers"]:
            report += f"\n• {p['symbol']}: {'+' if p['unrealized_pnl_pct'] >= 0 else ''}{p['unrealized_pnl_pct']:.1f}%"

    # Top losers
    if positions.get("top_losers"):
        report += "\n\n📉 **本日のTop Losers**"
        for p in positions["top_losers"]:
            report += f"\n• {p['symbol']}: {'+' if p['unrealized_pnl_pct'] >= 0 else ''}{p['unrealized_pnl_pct']:.1f}%"

    # Today's exits
    if exits:
        report += f"\n\n🔻 **本日のエグジット（{len(exits)}件）**"
        for e in exits[:5]:
            report += f"\n• {e['symbol']}"
        if len(exits) > 5:
            report += f"\n• ...他{len(exits) - 5}件"
    else:
        report += "\n\n🔻 **本日のエグジット**: なし"

    report += "\n\n✅ レポート生成完了"

    return report


def send_report(paper: bool = True) -> None:
    """Send daily summary report via Slack."""
    report = build_report(paper)

    print(report)
    print("\n" + "=" * 50)

    try:
        notifier = create_notifier(platform="slack", fallback=True)
        notifier.send("📊 日次サマリーレポート", report, channel=None)
        logger.info("Daily summary report sent successfully")
    except Exception as e:
        logger.error(f"Failed to send report: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    paper = not args.live
    send_report(paper)


if __name__ == "__main__":
    main()
