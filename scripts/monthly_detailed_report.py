"""Monthly Detailed Report - Full performance analysis with Excel export.

Scheduled to run on the 1st of each month at 09:00 JST.
Generates detailed Excel/CSV reports with:
- Account history
- Trade history
- System performance breakdown
- PnL analysis

Usage:
    python scripts/monthly_detailed_report.py [--paper]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common import broker_alpaca as ba
from common.notifier import create_notifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results_csv"
REPORTS_DIR = ROOT / "reports"


def ensure_reports_dir():
    """Ensure reports directory exists."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_order_history(paper: bool = True) -> pd.DataFrame:
    """Get order history from Alpaca."""
    try:
        client = ba.get_client(paper=paper)

        # Get orders from the past month
        end_date = datetime.now()
        start_date = end_date - timedelta(days=31)

        orders = client.get_orders(
            status="all",
            after=start_date.isoformat(),
            until=end_date.isoformat(),
            limit=500,
        )

        records = []
        for order in orders:
            records.append(
                {
                    "order_id": getattr(order, "id", ""),
                    "symbol": getattr(order, "symbol", ""),
                    "side": str(getattr(order, "side", "")),
                    "qty": int(getattr(order, "qty", 0) or 0),
                    "filled_qty": int(getattr(order, "filled_qty", 0) or 0),
                    "order_type": str(getattr(order, "order_type", "")),
                    "status": str(getattr(order, "status", "")),
                    "submitted_at": str(getattr(order, "submitted_at", "")),
                    "filled_at": str(getattr(order, "filled_at", "")),
                    "filled_avg_price": float(
                        getattr(order, "filled_avg_price", 0) or 0
                    ),
                }
            )

        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"Failed to get order history: {e}")
        return pd.DataFrame()


def get_portfolio_history(paper: bool = True) -> pd.DataFrame:
    """Get portfolio history from Alpaca."""
    try:
        client = ba.get_client(paper=paper)

        # Get portfolio history for the past month
        history = client.get_portfolio_history(
            period="1M",
            timeframe="1D",
        )

        if history is None:
            return pd.DataFrame()

        timestamps = getattr(history, "timestamp", [])
        equity = getattr(history, "equity", [])
        profit_loss = getattr(history, "profit_loss", [])
        profit_loss_pct = getattr(history, "profit_loss_pct", [])

        records = []
        for i, ts in enumerate(timestamps):
            records.append(
                {
                    "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                    "equity": equity[i] if i < len(equity) else 0,
                    "profit_loss": profit_loss[i] if i < len(profit_loss) else 0,
                    "profit_loss_pct": (
                        profit_loss_pct[i] * 100 if i < len(profit_loss_pct) else 0
                    ),
                }
            )

        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"Failed to get portfolio history: {e}")
        return pd.DataFrame()


def get_trade_log(paper: bool = True) -> pd.DataFrame:
    """Get enhanced trade log with entry/exit details from Alpaca orders."""
    try:
        client = ba.get_client(paper=paper)

        # Get filled orders from the past month
        end_date = datetime.now()
        start_date = end_date - timedelta(days=31)

        orders = client.get_orders(
            status="all",
            after=start_date.isoformat(),
            until=end_date.isoformat(),
            limit=500,
        )

        # Group orders by symbol to match entries with exits
        symbol_orders: dict[str, list] = {}
        for order in orders:
            status = str(getattr(order, "status", "")).lower()
            if "filled" not in status:
                continue
            symbol = getattr(order, "symbol", "")
            if symbol not in symbol_orders:
                symbol_orders[symbol] = []
            symbol_orders[symbol].append(order)

        trades = []
        for symbol, sym_orders in symbol_orders.items():
            # Sort by filled_at
            sym_orders.sort(
                key=lambda o: str(getattr(o, "filled_at", "") or "")
            )

            # Match buys with sells
            entry = None
            for order in sym_orders:
                side = str(getattr(order, "side", "")).lower()

                if "buy" in side and entry is None:
                    entry = order
                elif "sell" in side and entry is not None:
                    # Create trade record
                    entry_date = str(getattr(entry, "filled_at", ""))[:10]
                    exit_date = str(getattr(order, "filled_at", ""))[:10]
                    entry_price = float(
                        getattr(entry, "filled_avg_price", 0) or 0
                    )
                    exit_price = float(
                        getattr(order, "filled_avg_price", 0) or 0
                    )
                    qty = int(getattr(entry, "filled_qty", 0) or 0)

                    # Calculate PnL
                    pnl = (exit_price - entry_price) * qty
                    pnl_pct = (
                        ((exit_price - entry_price) / entry_price * 100)
                        if entry_price > 0 else 0
                    )

                    # Calculate holding days
                    try:
                        entry_dt = datetime.fromisoformat(
                            entry_date.replace("Z", "")
                        )
                        exit_dt = datetime.fromisoformat(
                            exit_date.replace("Z", "")
                        )
                        holding_days = (exit_dt - entry_dt).days
                    except Exception:
                        holding_days = 0

                    trades.append({
                        "symbol": symbol,
                        "side": "Long",
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(exit_price, 2),
                        "quantity": qty,
                        "realized_pnl": round(pnl, 2),
                        "realized_pnl_pct": round(pnl_pct, 2),
                        "holding_days": holding_days,
                        "system": _get_system_for_symbol(symbol),
                    })
                    entry = None

        if trades:
            return pd.DataFrame(trades)

        # Fallback to sent markers if no Alpaca trades
        return _get_trade_log_from_markers()

    except Exception as e:
        logger.error(f"Failed to get trade log: {e}")
        return _get_trade_log_from_markers()


def _get_system_for_symbol(symbol: str) -> str:
    """Get system name for a symbol from symbol_system_map."""
    try:
        map_path = DATA_DIR / "symbol_system_map.json"
        if map_path.exists():
            with map_path.open("r", encoding="utf8") as f:
                mapping = json.load(f)
            return mapping.get(symbol, "unknown")
    except Exception:
        pass
    return "unknown"


def _get_trade_log_from_markers() -> pd.DataFrame:
    """Fallback: Get trade log from sent markers."""
    try:
        sent_path = DATA_DIR / "alpaca_sent_markers.json"
        if not sent_path.exists():
            return pd.DataFrame()

        with sent_path.open("r", encoding="utf8") as f:
            markers = json.load(f)

        records = []
        for key, value in markers.items():
            symbol = key.split("_")[0]
            when = value.get("when", "")
            records.append({
                "symbol": symbol,
                "side": "Long",
                "entry_date": "",
                "exit_date": when[:10] if when else "",
                "entry_price": 0,
                "exit_price": 0,
                "quantity": 0,
                "realized_pnl": 0,
                "realized_pnl_pct": 0,
                "holding_days": 0,
                "system": _get_system_for_symbol(symbol),
            })

        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"Failed to get trade log from markers: {e}")
        return pd.DataFrame()


def calculate_trade_metrics(trades_df: pd.DataFrame) -> dict[str, Any]:
    """Calculate trading performance metrics."""
    if trades_df.empty:
        return {}

    total_trades = len(trades_df)

    # Win/Loss
    if "realized_pnl" in trades_df.columns:
        winners = trades_df[trades_df["realized_pnl"] > 0]
        losers = trades_df[trades_df["realized_pnl"] < 0]
        win_count = len(winners)
        loss_count = len(losers)
        win_rate = win_count / total_trades * 100 if total_trades > 0 else 0

        total_pnl = trades_df["realized_pnl"].sum()
        avg_pnl = trades_df["realized_pnl"].mean()
        max_win = trades_df["realized_pnl"].max()
        max_loss = trades_df["realized_pnl"].min()

        gross_profit = winners["realized_pnl"].sum() if not winners.empty else 0
        gross_loss = abs(losers["realized_pnl"].sum()) if not losers.empty else 0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )
    else:
        win_rate = total_pnl = avg_pnl = max_win = max_loss = 0
        profit_factor = 0
        win_count = loss_count = 0

    # Holding period
    if "holding_days" in trades_df.columns:
        avg_holding = trades_df["holding_days"].mean()
    else:
        avg_holding = 0

    return {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate_pct": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "max_win": round(max_win, 2),
        "max_loss": round(max_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_holding_days": round(avg_holding, 1),
    }


def get_system_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Get performance breakdown by system."""
    if trades_df.empty or "system" not in trades_df.columns:
        return pd.DataFrame()

    systems = trades_df.groupby("system").agg({
        "symbol": "count",
        "realized_pnl": ["sum", "mean"],
        "realized_pnl_pct": "mean",
        "holding_days": "mean",
    }).reset_index()

    systems.columns = [
        "system", "trade_count", "total_pnl", "avg_pnl",
        "avg_pnl_pct", "avg_holding_days"
    ]

    # Calculate win rate per system
    win_rates = []
    for system in systems["system"]:
        sys_trades = trades_df[trades_df["system"] == system]
        wins = len(sys_trades[sys_trades["realized_pnl"] > 0])
        total = len(sys_trades)
        win_rates.append(wins / total * 100 if total > 0 else 0)

    systems["win_rate_pct"] = [round(r, 1) for r in win_rates]

    return systems.round(2)


def generate_dummy_trades() -> pd.DataFrame:
    """Generate dummy trade data for testing."""
    import random

    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN", "AMD"]
    systems = ["system3", "system5", "system6"]

    trades = []
    base_date = datetime.now() - timedelta(days=30)

    for i in range(15):
        symbol = random.choice(symbols)
        system = random.choice(systems)
        entry_date = base_date + timedelta(days=random.randint(0, 20))
        holding = random.randint(1, 10)
        exit_date = entry_date + timedelta(days=holding)

        entry_price = random.uniform(50, 500)
        pnl_pct = random.uniform(-15, 25)
        exit_price = entry_price * (1 + pnl_pct / 100)
        qty = random.randint(5, 50)
        pnl = (exit_price - entry_price) * qty

        trades.append({
            "symbol": symbol,
            "side": "Long",
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "quantity": qty,
            "realized_pnl": round(pnl, 2),
            "realized_pnl_pct": round(pnl_pct, 2),
            "holding_days": holding,
            "system": system,
        })

    return pd.DataFrame(trades)


def generate_monthly_report(paper: bool = True) -> dict[str, Path]:
    """Generate monthly detailed report files."""
    ensure_reports_dir()

    month_str = datetime.now().strftime("%Y%m")
    report_files = {}

    # 1. Portfolio History
    portfolio_df = get_portfolio_history(paper)
    if not portfolio_df.empty:
        filepath = REPORTS_DIR / f"portfolio_history_{month_str}.csv"
        portfolio_df.to_csv(filepath, index=False, encoding="utf-8-sig")
        report_files["portfolio_history"] = filepath
        logger.info(f"Saved: {filepath}")

    # 2. Order History
    orders_df = get_order_history(paper)
    if not orders_df.empty:
        filepath = REPORTS_DIR / f"order_history_{month_str}.csv"
        orders_df.to_csv(filepath, index=False, encoding="utf-8-sig")
        report_files["order_history"] = filepath
        logger.info(f"Saved: {filepath}")

    # 3. Trade Log (enhanced)
    trade_log_df = get_trade_log(paper)
    if not trade_log_df.empty:
        filepath = REPORTS_DIR / f"trade_log_{month_str}.csv"
        trade_log_df.to_csv(filepath, index=False, encoding="utf-8-sig")
        report_files["trade_log"] = filepath
        logger.info(f"Saved: {filepath}")

        # 4. Trade Metrics
        metrics = calculate_trade_metrics(trade_log_df)
        if metrics:
            metrics_df = pd.DataFrame([
                {"metric": k, "value": v} for k, v in metrics.items()
            ])
            filepath = REPORTS_DIR / f"trade_metrics_{month_str}.csv"
            metrics_df.to_csv(filepath, index=False, encoding="utf-8-sig")
            report_files["trade_metrics"] = filepath
            logger.info(f"Saved: {filepath}")

        # 5. System Performance
        sys_perf_df = get_system_performance(trade_log_df)
        if not sys_perf_df.empty:
            filepath = REPORTS_DIR / f"system_performance_{month_str}.csv"
            sys_perf_df.to_csv(filepath, index=False, encoding="utf-8-sig")
            report_files["system_performance"] = filepath
            logger.info(f"Saved: {filepath}")

    # 4. Summary Statistics
    summary_data = []

    if not portfolio_df.empty:
        summary_data.append(
            {
                "metric": "期間開始残高",
                "value": portfolio_df.iloc[0]["equity"] if len(portfolio_df) > 0 else 0,
            }
        )
        summary_data.append(
            {
                "metric": "期間終了残高",
                "value": (
                    portfolio_df.iloc[-1]["equity"] if len(portfolio_df) > 0 else 0
                ),
            }
        )
        total_pnl = portfolio_df["profit_loss"].sum()
        summary_data.append(
            {
                "metric": "期間合計PnL",
                "value": total_pnl,
            }
        )

    if not orders_df.empty:
        filled = orders_df[orders_df["status"].str.contains("filled", case=False)]
        summary_data.append(
            {
                "metric": "約定注文数",
                "value": len(filled),
            }
        )
        buy_orders = filled[filled["side"].str.contains("buy", case=False)]
        sell_orders = filled[filled["side"].str.contains("sell", case=False)]
        summary_data.append(
            {
                "metric": "買い注文数",
                "value": len(buy_orders),
            }
        )
        summary_data.append(
            {
                "metric": "売り注文数",
                "value": len(sell_orders),
            }
        )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        filepath = REPORTS_DIR / f"monthly_summary_{month_str}.csv"
        summary_df.to_csv(filepath, index=False, encoding="utf-8-sig")
        report_files["summary"] = filepath
        logger.info(f"Saved: {filepath}")

    # 5. Combined Excel (if openpyxl is available)
    try:
        excel_path = REPORTS_DIR / f"monthly_report_{month_str}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            if not portfolio_df.empty:
                portfolio_df.to_excel(
                    writer, sheet_name="Portfolio History", index=False
                )
            if not orders_df.empty:
                orders_df.to_excel(writer, sheet_name="Order History", index=False)
            if not trade_log_df.empty:
                trade_log_df.to_excel(writer, sheet_name="Trade Log", index=False)
            if summary_data:
                pd.DataFrame(summary_data).to_excel(
                    writer, sheet_name="Summary", index=False
                )
        report_files["excel"] = excel_path
        logger.info(f"Saved Excel: {excel_path}")
    except ImportError:
        logger.warning("openpyxl not installed, skipping Excel export")
    except Exception as e:
        logger.warning(f"Failed to create Excel: {e}")

    return report_files


def send_notification(report_files: dict[str, Path], paper: bool = True) -> None:
    """Send Slack notification with report summary and Google Drive links."""
    try:
        client = ba.get_client(paper=paper)
        account = client.get_account()
        equity = float(getattr(account, "equity", 0) or 0)
    except Exception:
        equity = 0

    month_str = datetime.now().strftime("%Y年%m月")

    # Try to upload to OneDrive first, then Google Drive
    cloud_links = {}
    cloud_type = ""

    # Try OneDrive (simpler, uses local sync)
    try:
        from common.onedrive_uploader import upload_to_onedrive, ONEDRIVE_AVAILABLE

        if ONEDRIVE_AVAILABLE:
            logger.info("Copying reports to OneDrive...")
            cloud_type = "OneDrive"

            for name, path in report_files.items():
                try:
                    onedrive_path = upload_to_onedrive(path)
                    cloud_links[name] = onedrive_path
                    logger.info(f"Copied {name} to OneDrive")
                except Exception as e:
                    logger.error(f"Failed to copy {name} to OneDrive: {e}")

    except ImportError:
        logger.debug("OneDrive uploader not available")

    # Fallback to Google Drive if OneDrive failed
    if not cloud_links:
        try:
            from common.google_drive_uploader import upload_to_drive, GDRIVE_AVAILABLE
            import os

            folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

            if GDRIVE_AVAILABLE and folder_id:
                logger.info("Uploading reports to Google Drive...")
                cloud_type = "Google Drive"

                for name, path in report_files.items():
                    try:
                        url = upload_to_drive(path, folder_id=folder_id)
                        cloud_links[name] = url
                        logger.info(f"Uploaded {name}: {url}")
                    except Exception as e:
                        logger.error(f"Failed to upload {name}: {e}")

        except ImportError:
            logger.debug("Google Drive uploader not available")


    # Build notification message
    if cloud_links:
        # Cloud storage links available
        links_list = "\n".join(
            [f"• {name}: {path}" for name, path in cloud_links.items()]
        )

        message = f"""📊 【月次詳細レポート】{month_str}

💰 **現在の口座残高**
${equity:,.2f}

📁 **{cloud_type}レポート**
{links_list}

✅ ファイルは自動的にクラウドに同期されます。
"""
    else:
        # Fallback to local file list
        files_list = "\n".join(
            [f"• {name}: {path.name}" for name, path in report_files.items()]
        )

        message = f"""📊 【月次詳細レポート】{month_str}

💰 **現在の口座残高**
${equity:,.2f}

📁 **生成されたレポート**
{files_list}

📂 **保存先**
`reports/` フォルダ
"""

    print(message)

    try:
        notifier = create_notifier(platform="slack", fallback=True)
        notifier.send("📊 月次詳細レポート生成完了", message, channel=None)
        logger.info("Notification sent")
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 50)
    print("📊 Monthly Detailed Report Generator")
    print("=" * 50)

    report_files = generate_monthly_report(paper=args.paper)

    print(f"\nGenerated {len(report_files)} report files:")
    for name, path in report_files.items():
        print(f"  • {name}: {path}")

    send_notification(report_files, paper=args.paper)


if __name__ == "__main__":
    main()
