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
from pathlib import Path

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


__all__ = ["load_entry_dates", "save_entry_dates", "days_held", "ENTRY_DATE_PATH"]
