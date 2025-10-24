"""Simple per-symbol trade entry cache utilities.

The cache stores entry date and entry price for each symbol so that when an
exit signal arrives later we can look up the corresponding entry information
and visualise profit on charts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.io_utils import write_json

# File used to persist trade entry information
TRADE_CACHE_PATH = Path("data/trade_cache.json")


def _load_cache(path: Path = TRADE_CACHE_PATH) -> dict[str, dict[str, Any]]:
    """Load trade entry cache from ``path``.

    Returns an empty dictionary if the file does not exist or cannot be
    parsed.
    """
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except Exception:
        return {}


def _save_cache(
    cache: dict[str, dict[str, Any]],
    path: Path = TRADE_CACHE_PATH,
) -> None:
    """Persist ``cache`` to ``path`` as UTF-8 encoded JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # use centralized helper to ensure UTF-8 sanitization
    write_json(path, cache, ensure_ascii=False, indent=2)


def store_entry(
    symbol: str,
    entry_date: str,
    entry_price: float,
    *,
    path: Path = TRADE_CACHE_PATH,
) -> None:
    """Store ``entry_date`` and ``entry_price`` for ``symbol``.

    Any previous entry information for ``symbol`` will be overwritten.
    """
    cache = _load_cache(path)
    cache[symbol] = {"entry_date": entry_date, "entry_price": entry_price}
    _save_cache(cache, path)


def pop_entry(symbol: str, *, path: Path = TRADE_CACHE_PATH) -> dict[str, Any] | None:
    """Retrieve and remove cached entry for ``symbol`` if present."""
    cache = _load_cache(path)
    info = cache.pop(symbol, None)
    _save_cache(cache, path)
    return info


__all__ = ["store_entry", "pop_entry", "TRADE_CACHE_PATH"]
