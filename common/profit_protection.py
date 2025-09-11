"""Utilities for profit protection judgement.

Functions in this module inspect current positions and determine
whether exit conditions are met.  System7's 70-day high rule is
implemented along with simplified profit-protection checks for
Systems2, 3, 5, and 6.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from common.data_loader import load_price


def is_new_70day_high(symbol: str) -> bool:
    """Return True if the latest high is a new 70-day high for *symbol*.

    Parameters
    ----------
    symbol : str
        Ticker symbol to evaluate.

    Returns
    -------
    bool
        ``True`` when the most recent high price is greater than or equal to
        the highest high over the last 70 trading days.  ``False`` is returned
        if the price data cannot be loaded or the condition is not met.
    """

    try:
        df = load_price(symbol, cache_profile="rolling")
        if len(df) < 70:
            return False
        highs = df["High"]
        max_70 = float(highs.tail(70).max())
        latest_high = float(highs.iloc[-1])
        return latest_high >= max_70
    except Exception:
        return False


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


def evaluate_positions(positions: Iterable[Any]) -> pd.DataFrame:
    """Evaluate profit protection rules for given positions.

    Parameters
    ----------
    positions : Iterable[Any]
        Sequence of position objects from Alpaca's API.

    Returns
    -------
    pd.DataFrame
        DataFrame containing symbol, quantity, current price and judgement text.
    """

    records: list[dict[str, str]] = []
    for pos in positions:
        symbol = getattr(pos, "symbol", "")
        qty = getattr(pos, "qty", "")
        current = getattr(pos, "current_price", "")
        side = getattr(pos, "side", "")
        plpc = float(getattr(pos, "unrealized_plpc", 0) or 0)
        held = _days_held(getattr(pos, "entry_date", None))
        judgement = "継続"

        if symbol.upper() == "SPY" and side.lower() == "short":
            if is_new_70day_high("SPY"):
                judgement = "70日高値更新→翌日寄りで手仕舞い"
            else:
                judgement = "継続"
        elif side.lower() == "short":  # System2/6
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
                "qty": qty,
                "current_price": current,
                "judgement": judgement,
            }
        )

    return pd.DataFrame(records)
