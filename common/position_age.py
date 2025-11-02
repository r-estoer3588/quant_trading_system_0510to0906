"""Utilities for tracking position entry dates and computing holding days.

The Alpaca `get_all_positions()` API does not include the date when a
position was opened. In order to evaluate rules such as "exit after N days"
we store the entry dates when submitting orders and later look them up when
rendering dashboards or notifications.

The entry dates are persisted in ``data/position_entry_dates.json`` with the
following format::

    {
        "AAPL": "2025-05-01",
        "MSFT": "2025-05-03"
    }

This module provides helpers to load/save that mapping and compute how many
days have passed since the entry date.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

# File used to persist symbol -> entry_date mappings
ENTRY_DATE_PATH = Path("data/position_entry_dates.json")


def load_entry_dates(path: Path = ENTRY_DATE_PATH) -> dict[str, str]:
    """Load symbol to entry-date mapping from ``path``.

    Returns an empty dict if the file does not exist or cannot be parsed.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_entry_dates(mapping: dict[str, str], path: Path = ENTRY_DATE_PATH) -> None:
    """Persist ``mapping`` to ``path`` in UTF-8 encoded JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")


def days_held(entry_date: str | None) -> int | None:
    """Return number of days from ``entry_date`` to today.

    ``entry_date`` may be ``None`` or an unparsable string, in which case
    ``None`` is returned.
    """
    if not entry_date:
        return None
    try:
        d = pd.to_datetime(entry_date)
        today = pd.Timestamp.now(tz=d.tz)
        return int((today.normalize() - d.normalize()).days)
    except Exception:
        return None


def fetch_entry_dates_from_alpaca(
    client: Any, symbols: Iterable[str]
) -> dict[str, pd.Timestamp]:
    """Fetch entry dates for ``symbols`` from Alpaca fill activities.

    Parameters
    ----------
    client : Any
        Alpaca REST client instance providing ``get_activities``.
    symbols : Iterable[str]
        Iterable of ticker symbols for which entry dates should be
        retrieved. Duplicates and falsy values are ignored.

    Returns
    -------
    dict[str, pd.Timestamp]
        Mapping of upper-cased symbol strings to the earliest fill
        timestamp returned by the API. Symbols that could not be
        resolved are omitted.
    """

    if client is None:
        return {}

    out: dict[str, pd.Timestamp] = {}
    seen: set[str] = set()
    for symbol in symbols:
        try:
            sym = str(symbol).upper()
        except Exception:
            continue
        if not sym or sym in seen:
            continue
        seen.add(sym)
        try:
            activities = client.get_activities(  # type: ignore[attr-defined]
                symbol=sym, activity_types="FILL"
            )
        except Exception:
            continue

        # Alpaca returns most recent first; normalize to oldest fill.
        try:
            sorted_acts = sorted(
                activities, key=lambda a: getattr(a, "transaction_time", "")
            )
        except Exception:
            sorted_acts = list(activities)

        for act in sorted_acts:
            ts = getattr(act, "transaction_time", None)
            if not ts:
                continue
            try:
                out[sym] = pd.Timestamp(ts)
                break
            except Exception:
                continue

    return out


__all__ = [
    "load_entry_dates",
    "save_entry_dates",
    "days_held",
    "fetch_entry_dates_from_alpaca",
    "ENTRY_DATE_PATH",
]
