"""System candidate generation utilities (Phase-A commonization).

This module provides small, focused helper functions that eliminate
code duplication across System1-7's generate_candidates functions.

Phase-A utilities (safe, minimal impact):
- set_diagnostics_after_ranking: Update diagnostics counters after final ranking
- normalize_candidates_by_date: Convert list-of-dicts to {date: {symbol: payload}}
- choose_mode_date_for_latest_only: Determine mode date from date_counter for alignment

These functions do NOT change strategy logic—they only reduce boilerplate
and ensure consistent diagnostics/normalization behavior.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def set_diagnostics_after_ranking(
    diagnostics: dict[str, Any],
    *,
    final_df: pd.DataFrame | None,
    ranking_source: str,
    count_column: str = "symbol",
) -> None:
    """Update diagnostics after final ranking is complete.

    Args:
        diagnostics: Diagnostics dictionary to update in-place
        final_df: Final ranked DataFrame (or None if empty)
        ranking_source: 'latest_only' or 'full_scan'
        count_column: Column name to count for ranked_top_n_count (default 'symbol')

    Updates:
        - diagnostics['ranked_top_n_count']: Number of final candidates
        - diagnostics['ranking_source']: Source mode
        - diagnostics['setup_predicate_count']: Mirrors ranked count if not already set
    """
    if final_df is None or final_df.empty:
        diagnostics["ranked_top_n_count"] = 0
    else:
        diagnostics["ranked_top_n_count"] = len(final_df)

    diagnostics["ranking_source"] = ranking_source

    # For systems using single-source-of-truth pattern (setup == ranked),
    # mirror the count if setup_predicate_count is still 0
    if diagnostics.get("setup_predicate_count", 0) == 0:
        diagnostics["setup_predicate_count"] = diagnostics["ranked_top_n_count"]


def normalize_candidates_by_date(
    candidates_by_date: dict[pd.Timestamp, list[dict[str, Any]]],
) -> dict[pd.Timestamp, dict[str, dict[str, Any]]]:
    """Convert {date: [records]} to {date: {symbol: payload}}.

    Args:
        candidates_by_date: Dictionary mapping dates to list of candidate records

    Returns:
        Normalized mapping: {date: {symbol: {field: value}}}

    Each record in the input lists must have a 'symbol' key.
    The output omits 'symbol' and 'date' keys from the payload.
    """
    normalized: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}

    for dt, recs in candidates_by_date.items():
        symbol_map: dict[str, dict[str, Any]] = {}
        for rec in recs:
            sym = rec.get("symbol")
            if not isinstance(sym, str) or not sym:
                continue
            payload = {k: v for k, v in rec.items() if k not in ("symbol", "date")}
            symbol_map[sym] = payload
        normalized[dt] = symbol_map

    return normalized


def normalize_dataframe_to_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> dict[pd.Timestamp, dict[str, dict[str, Any]]]:
    """Convert DataFrame to {date: {symbol: payload}} format.

    Args:
        df: DataFrame with date and symbol columns
        date_col: Name of date column (default 'date')
        symbol_col: Name of symbol column (default 'symbol')

    Returns:
        Normalized mapping: {date: {symbol: {field: value}}}
    """
    if df is None or df.empty:
        return {}

    by_date: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}

    for dt_raw, sub in df.groupby(date_col):
        dt = pd.Timestamp(str(dt_raw))
        symbol_map: dict[str, dict[str, Any]] = {}
        for rec in sub.to_dict("records"):
            sym = rec.get(symbol_col)
            if not sym:
                continue
            payload: dict[str, Any] = {
                str(k): v for k, v in rec.items() if k not in (symbol_col, date_col)
            }
            symbol_map[str(sym)] = payload
        by_date[dt] = symbol_map

    return by_date


def choose_mode_date_for_latest_only(
    date_counter: dict[pd.Timestamp, int],
) -> pd.Timestamp | None:
    """Determine the mode (most frequent) date from a counter dict.

    Args:
        date_counter: Dictionary mapping dates to occurrence counts

    Returns:
        The date with the highest count, or None if empty

    This is used in latest_only paths to align all symbols to the same
    final date, providing resilience against missing data for some symbols.
    """
    if not date_counter:
        return None

    try:
        return max(date_counter.items(), key=lambda kv: kv[1])[0]
    except Exception:
        return None


__all__ = [
    "set_diagnostics_after_ranking",
    "normalize_candidates_by_date",
    "normalize_dataframe_to_by_date",
    "choose_mode_date_for_latest_only",
]
