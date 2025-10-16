"""Helpers to normalize candidate schemas across strategies.

This project historically returned two shapes from core/strategy layers:

- dict[pd.Timestamp, list[dict]]: per-date list of candidate records
- dict[pd.Timestamp, dict[str, dict]]: per-date mapping of symbol -> payload

Consumers increasingly expect the list-of-dicts shape. This module provides
helpers to normalize the per-date dict-of-dicts into a list-of-dicts, while
keeping ordering predictable:

1) If payload contains 'rank', sort ascending by rank
2) Else if payload contains 'return_6d', sort descending by return_6d
3) Else stable symbol order (dict insertion order) as a fallback

Notes:
- We do not touch records already in list form
- We ensure 'symbol' and 'entry_date' keys are present in each record
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd


def _to_timestamp_safe(dt: Any) -> pd.Timestamp:
    """Safely convert various date-like inputs to pd.Timestamp.

    If conversion fails, returns a normalized Timestamp built from str(dt),
    which will likely be NaT on invalid inputs; downstream callers should
    treat such records conservatively.
    """
    try:
        ts = pd.Timestamp(dt)
    except Exception:
        try:
            ts = pd.to_datetime(str(dt), errors="coerce")  # type: ignore[assignment]
        except Exception:
            ts = pd.NaT  # type: ignore[assignment]
    if isinstance(ts, pd.Timestamp):
        try:
            return ts.normalize()
        except Exception:
            return ts
    # Fallback: always return a valid pd.Timestamp (NaT)
    return pd.Timestamp("NaT")


def _sorted_items_by_priority(d: Dict[str, Dict[str, Any]]) -> Iterable[tuple[str, Dict[str, Any]]]:
    """Yield (symbol, payload) pairs in preferred order.

    Priority:
    - rank ascending if present
    - else return_6d descending if present
    - else insertion order (Python 3.7+ dict preserves insertion order)
    """
    # rank available? sort ascending
    if any(isinstance(v, dict) and "rank" in v for v in d.values()):
        try:
            return sorted(d.items(), key=lambda kv: (float(kv[1].get("rank", float("inf")))))
        except Exception:
            # fall through to return_6d ordering
            pass
    # return_6d available? sort descending
    if any(isinstance(v, dict) and "return_6d" in v for v in d.values()):
        try:
            return sorted(
                d.items(),
                key=lambda kv: (float(kv[1].get("return_6d", float("nan")))),
                reverse=True,
            )
        except Exception:
            pass
    # fallback: as-is
    return d.items()


def normalize_candidates_to_list(
    candidates_by_date: dict,
) -> dict[pd.Timestamp, list[dict]]:
    """Normalize per-date candidates to list-of-dicts shape.

    Input may be:
    - dict[pd.Timestamp|date-like, list[dict]]
    - dict[pd.Timestamp|date-like, dict[str, dict]]

    Returns:
    - dict[pd.Timestamp, list[dict]] where each dict has at least
      {'symbol': str, 'entry_date': Timestamp}
    """
    if not isinstance(candidates_by_date, dict):
        return {}

    out: dict[pd.Timestamp, list[dict]] = {}
    for date_key, entries in candidates_by_date.items():
        ts = _to_timestamp_safe(date_key)
        if isinstance(entries, list):
            # assume already record-shaped; ensure entry_date exists
            normalized_list: list[dict] = []
            for rec in entries:
                if not isinstance(rec, dict):
                    continue
                if "entry_date" not in rec:
                    # do not mutate original
                    tmp = dict(rec)
                    tmp["entry_date"] = ts
                    normalized_list.append(tmp)
                else:
                    normalized_list.append(rec)
            out[ts] = normalized_list
            continue

        # dict-of-dicts: symbol -> payload
        if isinstance(entries, dict):
            bucket: list[dict] = []
            for sym, payload in _sorted_items_by_priority(entries):
                if not isinstance(sym, str) or not isinstance(payload, dict):
                    continue
                rec = {"symbol": sym, **payload}
                # prefer existing entry_date in payload, else use the dict key (date)
                if "entry_date" not in rec or rec["entry_date"] is None:
                    rec["entry_date"] = ts
                else:
                    try:
                        rec["entry_date"] = _to_timestamp_safe(rec["entry_date"])
                    except Exception:
                        rec["entry_date"] = ts
                bucket.append(rec)
            out[ts] = bucket
            continue

        # unknown type -> skip
    return out


__all__ = [
    "normalize_candidates_to_list",
]
