"""Utilities for profit protection judgement.

Functions in this module inspect current positions and determine
whether exit conditions are met.  System7's 70-day high rule is
implemented along with simplified profit-protection checks for
Systems2, 3, 5, and 6.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common.data_loader import load_price


_SYMBOL_SYSTEM_MAP_PATH = Path("data/symbol_system_map.json")


def _load_symbol_system_map() -> dict[str, str]:
    """Load persisted symbol→system mapping.

    Returns an empty dictionary when the mapping file is missing or invalid.
    """

    try:
        data = json.loads(_SYMBOL_SYSTEM_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    mapping: dict[str, str] = {}
    for key, value in data.items():
        try:
            symbol = str(key).upper()
            system = str(value).strip()
        except Exception:
            continue
        if not symbol:
            continue
        mapping[symbol] = system.lower()
    return mapping


def is_new_70day_high(symbol: str = "SPY") -> bool | None:
    """Return whether ``symbol`` made a new 70-day high.

    Parameters
    ----------
    symbol : str, default "SPY"
        Ticker symbol to check.

    Returns
    -------
    bool | None
        ``True`` if the latest ``High`` is greater than or equal to the
        rolling 70-day maximum, ``False`` if not. ``None`` is returned when
        price data could not be loaded, allowing callers to treat the
        judgement as indeterminate instead of mistakenly continuing the
        position.
    """

    try:
        df = load_price(symbol, cache_profile="rolling")
    except Exception:
        return None
    try:
        df["max_70"] = df["High"].rolling(window=70).max()
        latest = df.iloc[-1]
        return float(latest["High"]) >= float(latest["max_70"])
    except Exception:
        return None


def _days_held(entry_date: Any) -> int | None:
    """Compute days held from entry_date to today."""

    if not entry_date:
        return None
    try:
        entry = pd.to_datetime(entry_date)
        today = pd.Timestamp.utcnow().normalize()
        return int((today - entry.normalize()).days)
    except Exception:
        return None


def _format_entry_date(entry_date: Any) -> str:
    """Return ``entry_date`` formatted as YYYY-MM-DD when possible."""

    if not entry_date:
        return ""
    try:
        dt = pd.to_datetime(entry_date)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(entry_date)


def _next_action_hint(
    system: str, side_lower: str, held: int | None, plpc: float
) -> str:
    """Build a short description about the next likely exit trigger."""

    system_norm = (system or "").lower()
    if system_norm == "system7":
        return "70日高値を更新すると翌日寄りで手仕舞い"

    hints: list[str] = []
    if side_lower == "short":
        if plpc < 0.05:
            hints.append("含み益5%で翌日大引け手仕舞い")
        if plpc < 0.04:
            hints.append("含み益4%で翌日大引け手仕舞い")
        if held is not None:
            if held < 2:
                hints.append("2日経過で大引け手仕舞い")
            if held < 3:
                hints.append("3日経過で大引け手仕舞い")
    elif side_lower == "long":
        if plpc < 0.04:
            hints.append("含み益4%で翌日大引け手仕舞い")
        if held is not None:
            if held < 3:
                hints.append("3日経過で翌日大引け手仕舞い")
            if held < 6:
                hints.append("6日経過で翌日寄り手仕舞い")

    return " / ".join(hints)


def _rule_summary(system: str, side_lower: str) -> str:
    """Return a high level summary of profit/loss handling for the system."""

    system_norm = (system or "").lower()
    if system_norm == "system7":
        return "SPYヘッジ: 70日高値で翌日寄り決済"
    if side_lower == "short":
        return "含み益4%-5%または2-3日経過で手仕舞い"
    if side_lower == "long":
        return "含み益4%または3・6日経過で手仕舞い"
    return "含み益と経過日数で手仕舞いを判断"


def _position_close_price(position: Any) -> float | str:
    """Return the last-day close for ``position`` when available."""

    for attr in ("lastday_price", "current_price"):
        candidate = getattr(position, attr, None)
        if candidate in (None, ""):
            continue
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    fallback = getattr(position, "current_price", None)
    if fallback in (None, ""):
        return ""
    return fallback


def evaluate_positions(positions: Iterable[Any]) -> pd.DataFrame:
    """Evaluate profit protection rules for given positions.

    Parameters
    ----------
    positions : Iterable[Any]
        Sequence of position objects from Alpaca's API.

    Returns
    -------
    pd.DataFrame
        DataFrame containing symbol, side, quantity, current price and judgement
        text.
    """

    raw_map = _load_symbol_system_map()
    symbol_system_map: dict[str, str] = {}
    for key, value in raw_map.items():
        try:
            sym_key = str(key).upper()
            sys_val = str(value).strip().lower()
        except Exception:
            continue
        if not sym_key:
            continue
        symbol_system_map[sym_key] = sys_val
    records: list[dict[str, str]] = []
    for pos in positions:
        symbol_raw = getattr(pos, "symbol", "")
        symbol = "" if symbol_raw in (None, "") else str(symbol_raw)
        symbol_key = symbol.upper()
        qty = getattr(pos, "qty", "")
        current = _position_close_price(pos)
        side = getattr(pos, "side", "")
        side_lower = str(side).lower()
        plpc = float(getattr(pos, "unrealized_plpc", 0) or 0)
        held = _days_held(getattr(pos, "entry_date", None))
        entry_date = _format_entry_date(getattr(pos, "entry_date", None))
        system = symbol_system_map.get(symbol_key, "")
        if not system and symbol_key == "SPY" and side_lower == "short":
            system = "system7"
        judgement = "継続"

        if symbol_key == "SPY" and side_lower == "short":
            high_check = is_new_70day_high(symbol)
            if high_check is True:
                judgement = "70日高値更新→翌日寄りで手仕舞い"
            elif high_check is None:
                judgement = "判定失敗"
        elif side_lower == "short":  # System2/6
            if plpc >= 0.05:
                judgement = "5%利益→翌日大引けで手仕舞い"
            elif held is not None and held >= 3:
                judgement = "3日経過→大引けで手仕舞い"
            elif plpc >= 0.04:
                judgement = "4%利益→翌日大引けで手仕舞い"
            elif held is not None and held >= 2:
                judgement = "2日経過→大引けで手仕舞い"
        else:  # long positions System3/5
            if plpc >= 0.04:
                judgement = "4%利益→翌日大引けで手仕舞い"
            elif held is not None and held >= 6:
                judgement = "6日経過→翌日寄りで手仕舞い"
            elif held is not None and held >= 3:
                judgement = "3日経過→翌日大引けで手仕舞い"

        records.append(
            {
                "symbol": symbol,
                "system": system,
                "side": side,
                "qty": qty,
                "current_price": current,
                "avg_entry_price": getattr(pos, "avg_entry_price", ""),
                "entry_date": entry_date,
                "holding_days": held,
                "unrealized_pl": getattr(pos, "unrealized_pl", ""),
                "unrealized_plpc_percent": round(plpc * 100, 2),
                "judgement": judgement,
                "next_action": _next_action_hint(system, side_lower, held, plpc),
                "rule_summary": _rule_summary(system, side_lower),
            }
        )

    return pd.DataFrame(records)
