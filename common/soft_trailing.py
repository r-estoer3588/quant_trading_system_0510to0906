"""Intraday soft trailing protection for legacy fractional S1/S4 Paper positions.

This module is the churn-free remediation path for positions that Alpaca cannot
protect with broker-resident stop/trailing orders because their quantity is
fractional.  New Paper entries are whole-share-only and therefore remain on the
native trailing-stop path; this monitor deliberately ignores whole-share
positions to avoid racing Alpaca's native protection.

Safety contract
---------------
* Paper only. ``paper=False`` is rejected and ``assert_paper_env`` is enforced.
* No blanket cancel/replace. The only write-side broker action is
  ``close_position(symbol)`` after a measured S1/S4 trailing threshold breach.
* S1/S4 percentages come from ``SYSTEM_TRADE_RULES`` (25% / 20%), never from a
  duplicated literal policy.
* HWM is persisted in ``data/trailing_stops.json`` and bootstrapped from cached
  daily highs since entry when available, so pre-existing fractional positions
  do not start with an entry-only pseudo-HWM.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

from common import broker_alpaca as ba
from common.alpaca_trading import (
    assert_paper_env,
    parse_entry_date_from_client_order_id,
    parse_system_from_client_order_id,
)
from common.position_tracker import load_tracker
from common.symbol_map import load_symbol_system_map
from common.trade_management import SYSTEM_TRADE_RULES

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "trailing_stops.json"
STATUS_DIR = ROOT / "results_csv"
LOG_DIR = ROOT / "logs"
PROTECTED_SYSTEMS = frozenset({"system1", "system4"})
_QTY_EPS = 1e-6


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower().split(".")[-1]


def is_fractional_qty(qty: float) -> bool:
    """Return True only when qty materially differs from a whole share."""
    return abs(abs(qty) - round(abs(qty))) > _QTY_EPS


def trailing_threshold(*, side: str, hwm: float, trail_pct: float) -> float:
    """Pure HWM ratchet threshold. Long S1/S4 are HWM*(1-trail_pct)."""
    if hwm <= 0 or not (0 < trail_pct < 1):
        raise ValueError("hwm must be positive and trail_pct must be between 0 and 1")
    side_n = side.strip().lower()
    if side_n == "long":
        return hwm * (1.0 - trail_pct)
    if side_n == "short":
        return hwm * (1.0 + trail_pct)
    raise ValueError(f"unsupported side: {side}")


def update_hwm(
    *, side: str, previous: float | None, entry: float, current: float
) -> float:
    """Pure monotone ratchet update."""
    vals = [v for v in (previous, entry, current) if v is not None and v > 0]
    if not vals:
        raise ValueError("at least one positive price is required")
    return max(vals) if side == "long" else min(vals)


def _load_state(path: Path = STATE_PATH) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state: dict[str, dict[str, Any]], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_status(rows: list[dict[str, Any]], *, stamp: datetime) -> Path:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATUS_DIR / f"soft_trailing_status_{stamp:%Y%m%d}.json"
    payload = {
        "schema": "soft_trailing_status/v1",
        "generated_at": stamp.isoformat(),
        "paper": True,
        "systems": sorted(PROTECTED_SYSTEMS),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _order_system_index(client: Any) -> dict[str, dict[str, str | None]]:
    """Best-effort read-only system/entry index from Alpaca entry order COIDs."""
    out: dict[str, dict[str, str | None]] = {}
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500)
        )
    except Exception as exc:
        logger.warning("soft trailing: unable to read order attribution: %s", exc)
        return out

    for order in orders or []:
        sym = str(getattr(order, "symbol", "") or "").upper()
        if not sym or sym in out:
            continue
        coid = str(getattr(order, "client_order_id", "") or "")
        system = parse_system_from_client_order_id(coid)
        if not system:
            continue
        filled_at = getattr(order, "filled_at", None)
        entry_date = (
            str(filled_at)[:10]
            if filled_at
            else parse_entry_date_from_client_order_id(coid)
        )
        out[sym] = {"system": str(system).lower(), "entry_date": entry_date}
    return out


def _resolve_position_meta(
    symbol: str,
    *,
    orders: dict[str, dict[str, str | None]],
    tracker: dict[str, Any],
    symbol_map: dict[str, str],
) -> tuple[str | None, str | None]:
    """Match exporter attribution priority: Alpaca COID -> tracker -> static map."""
    sym = symbol.upper()
    order_row = orders.get(sym) or {}
    tr = tracker.get(sym) or tracker.get(sym.lower()) or {}
    system = order_row.get("system")
    entry_date = order_row.get("entry_date")
    if not system and isinstance(tr, dict):
        system = str(tr.get("system") or "").lower() or None
    if not entry_date and isinstance(tr, dict):
        entry_date = str(tr.get("entry_date") or "")[:10] or None
    if not system:
        system = symbol_map.get(sym.lower())
    return (str(system).lower() if system else None, entry_date)


def _entry_price_from_tracker(
    symbol: str, tracker: dict[str, Any], fallback: float
) -> float:
    row = tracker.get(symbol.upper()) or tracker.get(symbol.lower()) or {}
    if isinstance(row, dict):
        v = _f(row.get("entry_price"))
        if v and v > 0:
            return v
    return fallback


def _bootstrap_high_from_cache(
    symbol: str, entry_date: str | None, entry_price: float
) -> float:
    """Best-effort historical long HWM from rolling CSV daily High values."""
    best = entry_price
    path = ROOT / "data_cache" / "rolling" / f"{symbol.upper()}.csv"
    if not entry_date or not path.exists():
        return best
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                return best
            lower = {name.lower(): name for name in reader.fieldnames}
            date_col = next(
                (lower[k] for k in ("date", "datetime", "timestamp") if k in lower),
                None,
            )
            high_col = lower.get("high")
            if not date_col or not high_col:
                return best
            for row in reader:
                day = str(row.get(date_col) or "")[:10]
                if not day or day < entry_date[:10]:
                    continue
                high = _f(row.get(high_col))
                if high and high > best:
                    best = high
    except Exception as exc:
        logger.warning("soft trailing: HWM bootstrap failed for %s: %s", symbol, exc)
    return best


def _same_position(
    prev: dict[str, Any], *, system: str, entry_date: str | None, entry_price: float
) -> bool:
    if not prev or str(prev.get("system") or "").lower() != system:
        return False
    old_entry = _f(prev.get("entry_price"))
    if old_entry is None or abs(old_entry - entry_price) > max(
        0.01, entry_price * 1e-4
    ):
        return False
    old_date = str(prev.get("entry_date") or "")[:10] or None
    if old_date and entry_date and old_date != entry_date[:10]:
        return False
    return True


def run_soft_trailing(
    *,
    paper: bool = True,
    dry_run: bool = False,
    client: Any | None = None,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate and, on breach, market-close fractional S1/S4 Paper positions.

    The function is safe to call every 30 minutes. Whole-share S1/S4 positions
    are skipped because those are protected by Alpaca native trailing orders.
    """
    if not paper:
        raise RuntimeError("soft trailing monitor is Paper-only")
    assert_paper_env()
    client = client or ba.get_client(paper=True)
    state_file = state_path or STATE_PATH
    stamp = now or datetime.now(timezone.utc)

    try:
        clock = client.get_clock()
        market_open = bool(getattr(clock, "is_open", False))
    except Exception as exc:
        logger.error("soft trailing: broker clock unavailable; fail closed: %s", exc)
        return {
            "paper": True,
            "market_open": None,
            "checked": 0,
            "protected": 0,
            "closed": 0,
            "error": str(exc),
        }
    if not market_open:
        return {
            "paper": True,
            "market_open": False,
            "checked": 0,
            "protected": 0,
            "closed": 0,
            "rows": [],
        }

    positions = list(client.get_all_positions() or [])
    tracker = load_tracker() or {}
    symbol_map = load_symbol_system_map() or {}
    orders = _order_system_index(client)
    state = _load_state(state_file)
    rows: list[dict[str, Any]] = []
    closed = 0

    for pos in positions:
        symbol = str(getattr(pos, "symbol", "") or "").upper()
        qty = _f(getattr(pos, "qty", None))
        if not symbol or qty is None or qty == 0 or not is_fractional_qty(qty):
            continue

        system, entry_date = _resolve_position_meta(
            symbol, orders=orders, tracker=tracker, symbol_map=symbol_map
        )
        if system not in PROTECTED_SYSTEMS:
            continue
        rules = SYSTEM_TRADE_RULES.get(system)
        if rules is None or not getattr(rules, "use_trailing_stop", False):
            continue
        trail_pct = float(getattr(rules, "trailing_stop_pct", 0.0) or 0.0)
        if not (0 < trail_pct < 1):
            continue

        side = _enum_value(getattr(pos, "side", None))
        if side not in {"long", "short"}:
            side = "long" if qty > 0 else "short"
        # S1/S4 are long by contract. Fail closed instead of managing an anomalous short.
        if side != "long":
            logger.error(
                "soft trailing: %s %s has anomalous side=%s; skipped",
                system,
                symbol,
                side,
            )
            continue

        avg_entry = _f(getattr(pos, "avg_entry_price", None)) or 0.0
        current = _f(getattr(pos, "current_price", None))
        if current is None:
            mv = _f(getattr(pos, "market_value", None))
            current = abs(mv / qty) if mv is not None and qty else None
        if avg_entry <= 0 or current is None or current <= 0:
            logger.warning(
                "soft trailing: %s missing measured entry/current price; skipped",
                symbol,
            )
            continue

        entry_price = _entry_price_from_tracker(symbol, tracker, avg_entry)
        prev = state.get(symbol) if isinstance(state.get(symbol), dict) else {}
        same = _same_position(
            prev, system=system, entry_date=entry_date, entry_price=entry_price
        )
        previous_hwm = _f(prev.get("highest_price")) if same else None
        bootstrap_hwm = _bootstrap_high_from_cache(symbol, entry_date, entry_price)
        hwm = update_hwm(
            side="long",
            previous=max(v for v in (previous_hwm, bootstrap_hwm) if v is not None),
            entry=entry_price,
            current=current,
        )
        stop = trailing_threshold(side="long", hwm=hwm, trail_pct=trail_pct)
        breached = current <= stop

        state[symbol] = {
            "system": system,
            "entry_date": entry_date,
            "entry_price": round(entry_price, 6),
            "highest_price": round(hwm, 6),
            "trailing_stop_price": round(stop, 6),
            "trailing_stop_pct": trail_pct,
            "last_price": round(current, 6),
            "last_update": stamp.isoformat(),
            "source": "soft_market_monitor",
        }
        row: dict[str, Any] = {
            "symbol": symbol,
            "system": system,
            "qty": round(qty, 6),
            "entry_date": entry_date,
            "entry_price": round(entry_price, 6),
            "current_price": round(current, 6),
            "hwm": round(hwm, 6),
            "trail_pct": trail_pct,
            "stop_price": round(stop, 6),
            "breached": breached,
            "action": "none",
            "order_id": None,
            "order_status": None,
        }

        if breached:
            if dry_run:
                row["action"] = "would_close_market"
            else:
                try:
                    order = client.close_position(symbol)
                    row["action"] = "close_market_submitted"
                    row["order_id"] = str(getattr(order, "id", "") or "") or None
                    row["order_status"] = (
                        _enum_value(getattr(order, "status", None)) or None
                    )
                    closed += 1
                    _append_jsonl(
                        {"timestamp": stamp.isoformat(), "paper": True, **row},
                        LOG_DIR / f"soft_trailing_exits_{stamp:%Y%m%d}.jsonl",
                    )
                except Exception as exc:
                    row["action"] = "close_market_failed"
                    row["error"] = str(exc)
                    logger.exception("soft trailing close failed for %s", symbol)
        rows.append(row)

    _save_state(state, state_file)
    status_path = _write_status(rows, stamp=stamp)
    result = {
        "paper": True,
        "market_open": True,
        "checked": len(positions),
        "protected": len(rows),
        "closed": closed,
        "dry_run": dry_run,
        "status_path": str(status_path),
        "rows": rows,
    }
    logger.info(
        "soft trailing: positions=%d fractional_s1_s4=%d closes=%d dry_run=%s",
        len(positions),
        len(rows),
        closed,
        dry_run,
    )
    return result
