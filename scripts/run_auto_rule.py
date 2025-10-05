"""Run auto-rule outside Streamlit.

This script loads settings and auto-rule config from the same files used by
`app_alpaca_dashboard.py`, loads current positions using the broker wrapper,
and submits exit orders via `submit_exit_orders_df`.

Usage:
    python scripts/run_auto_rule.py --paper --dry-run

Notes:
- Ensure environment variables (ALPACA keys, SLACK/Discord tokens) are available to the process.
- Recommended to run under the same venv as the app.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common import broker_alpaca as ba
from common.alpaca_order import submit_exit_orders_df
from common.notifier import Notifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SENT_PATH = DATA_DIR / "alpaca_sent_markers.json"
CONFIG_PATH = DATA_DIR / "auto_rule_config.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_auto_rule")


def load_json(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_json(path: Path, d: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("failed to save json")


def today_key_for(sym: str) -> str:
    return f"{sym}_today_close_{datetime.now().date().isoformat()}"


def load_sent_markers() -> dict[str, Any]:
    return load_json(SENT_PATH)


def mark_sent(sym: str, markers: dict[str, Any]) -> None:
    markers[today_key_for(sym)] = {"when": datetime.now().isoformat()}


def build_auto_rows(cfg: dict[str, Any], markers: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # fetch positions via Alpaca client
    client = ba.get_client()
    try:
        positions = client.get_all_positions()
    except Exception:
        logger.exception("failed to fetch positions")
        return rows
    # normalize to list of dicts similar to UI DataFrame
    records: list[dict[str, Any]] = []
    for p in positions:
        try:
            sym = getattr(p, "symbol", None) or getattr(p, "symbol_raw", None)
            qty = int(getattr(p, "qty", 0) or 0)
            avg = float(getattr(p, "avg_entry_price", 0.0) or 0.0)
            cur = float(getattr(p, "market_value", 0.0) or 0.0)
            # approximate pnl pct if price available
            try:
                price = float(getattr(p, "current_price", 0.0) or 0.0)
                pnl_pct = ((price - avg) / avg * 100.0) if avg else 0.0
            except Exception:
                pnl_pct = 0.0
            records.append(
                {
                    "symbol": sym,
                    "数量": qty,
                    "平均取得単価": avg,
                    "現在値": cur,
                    "損益率(%)": pnl_pct,
                    "side": getattr(p, "side", ""),
                }
            )
        except Exception:
            continue

    pos_df = pd.DataFrame(records)
    if pos_df is None or pos_df.empty:
        return rows
    for _, r in pos_df.iterrows():
        try:
            sym = str(r.get("symbol", "")).upper()
            if not sym:
                continue
            system_name = str(r.get("システム", "")).strip() or "unknown"
            c = cfg.get(system_name, {})
            threshold = float(c.get("pnl_threshold", -20.0))
            partial_pct = int(c.get("partial_pct", 100))
            pnl_pct = float(r.get("損益率(%)", 0.0) or 0.0)
            # Note: This flag is likely always False when run from this script.
            limit_reached = bool(r.get("_limit_reached"))
            if limit_reached or pnl_pct <= threshold:
                key = today_key_for(sym)
                if key in markers:
                    logger.info("skip %s already sent today", sym)
                    continue
                qty = int(r.get("数量") or r.get("qty") or 0)
                if qty <= 0:
                    continue
                apply_qty = max(1, int(qty * partial_pct / 100))
                rows.append(
                    {
                        "symbol": sym,
                        "qty": apply_qty,
                        "position_side": r.get("side") or r.get("position_side") or "",
                        "system": system_name,
                        "when": "today_close",
                    }
                )
        except Exception:
            logger.exception("failed to evaluate row")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", help="use paper trading mode")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not submit orders, only simulate",
    )
    args = parser.parse_args()

    cfg = load_json(CONFIG_PATH)
    markers = load_sent_markers()
    rows = build_auto_rows(cfg, markers)
    if not rows:
        logger.info("no candidates for auto-rule")
        return
    df = pd.DataFrame(rows)
    logger.info("candidates: %s", ", ".join(r["symbol"] for r in rows))

    if args.dry_run:
        logger.info("dry-run enabled, not submitting orders")
        return

    try:
        res = submit_exit_orders_df(df, paper=args.paper, tif="CLS", notify=True)
        logger.info("submitted %d orders", len(res))
        for r in rows:
            mark_sent(r["symbol"], markers)
        save_json(SENT_PATH, markers)
        try:
            nd = load_json(DATA_DIR / "notify_settings.json")
            notifier = Notifier(
                platform=nd.get("platform", "auto"),
                webhook_url=nd.get("webhook_url"),
            )
            notifier.send(
                "自動ルール: まとめて決済実行",
                "送信銘柄: " + ", ".join(r["symbol"] for r in rows),
            )
        except Exception:
            logger.exception("notify failed")
    except Exception:
        logger.exception("submission failed")


if __name__ == "__main__":
    main()
