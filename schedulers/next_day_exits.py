from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import broker_alpaca as ba
from common.notifier import Notifier

"""
Simple scheduler to execute planned exit orders at next day's open/close.

Usage:
    - This script reads a JSONL plan file produced by the UI containing lines of
        {symbol, qty, position_side, system, when} where when is one of:
            - "tomorrow_open"
            - "tomorrow_close"

    - Run this script around market open (09:30 ET) and/or before market close to
        submit the corresponding orders via Alpaca.

Plan file path: data/planned_exits.jsonl

Note:
    - This tool assumes environment variables for Alpaca are set.
    - It will ignore entries whose time window does not match (e.g., running the
        "open" window will skip "tomorrow_close" records).
"""

PLAN_PATH = Path("data/planned_exits.jsonl")


def _load_plans() -> list[dict[str, Any]]:
    if not PLAN_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in PLAN_PATH.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _save_plans(plans: list[dict[str, Any]]) -> None:
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLAN_PATH.open("w", encoding="utf-8") as f:
        for rec in plans:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _is_market_open_window(now: pd.Timestamp) -> bool:
    # naive check: 09:30-09:45 ET
    t = now.tz_convert("America/New_York").time()
    return (t >= pd.Timestamp("09:30", tz="America/New_York").time()) and (
        t <= pd.Timestamp("09:45", tz="America/New_York").time()
    )


def _is_market_close_window(now: pd.Timestamp) -> bool:
    # naive check: 15:50-16:00 ET
    t = now.tz_convert("America/New_York").time()
    return (t >= pd.Timestamp("15:50", tz="America/New_York").time()) and (
        t <= pd.Timestamp("16:00", tz="America/New_York").time()
    )


def submit_planned_exits(window: str, dry_run: bool = False) -> pd.DataFrame:
    plans = _load_plans()
    if not plans:
        return pd.DataFrame()
    try:
        client = ba.get_client(paper=None)
    except Exception:
        client = None
    rows: list[dict[str, Any]] = []
    for rec in plans:
        when = str(rec.get("when", "")).lower()
        if window == "open" and when != "tomorrow_open":
            continue
        if window == "close" and when != "tomorrow_close":
            continue
        sym = str(rec.get("symbol"))
        qty = int(rec.get("qty") or 0)
        pos_side = str(rec.get("position_side", "")).lower()
        system = str(rec.get("system", ""))
        if not sym or qty <= 0:
            continue
        side = "sell" if pos_side == "long" else "buy"
        order_type = "market"
        tif = "OPG" if window == "open" else "CLS"
        try:
            if dry_run:
                rows.append(
                    {
                        "symbol": sym,
                        "qty": qty,
                        "side": side,
                        "when": when,
                        "system": system,
                        "dry_run": True,
                    }
                )
            else:
                if client is None:
                    raise RuntimeError("Alpaca client not available")
                order = ba.submit_order_with_retry(
                    client,
                    sym,
                    qty,
                    side=side,
                    order_type=order_type,
                    time_in_force=tif,
                    retries=2,
                    backoff_seconds=0.5,
                    rate_limit_seconds=0.2,
                    log_callback=None,
                )
                rows.append(
                    {
                        "symbol": sym,
                        "qty": qty,
                        "side": side,
                        "when": when,
                        "system": system,
                        "order_id": getattr(order, "id", None),
                        "status": getattr(order, "status", None),
                    }
                )
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "side": side,
                    "when": when,
                    "system": system,
                    "error": str(e),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        try:
            Notifier(platform="auto").send_trade_report(f"planned-exits-{window}", rows)
        except Exception:
            pass
    # Remove executed ones from plan（dry_run のときは削除しない）
    if not dry_run:
        target = "tomorrow_open" if window == "open" else "tomorrow_close"
        remain = [r for r in plans if str(r.get("when", "")).lower() != target]
        _save_plans(remain)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["open", "close"], required=False)
    args = ap.parse_args()
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    window = args.window
    if not window:
        if _is_market_open_window(now):
            window = "open"
        elif _is_market_close_window(now):
            window = "close"
        else:
            print("Not in open/close window; specify --window.")
            return
    df = submit_planned_exits(window)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No planned exits to submit.")


if __name__ == "__main__":
    main()
