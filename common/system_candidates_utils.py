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


# ===============================
# Option-B helper (non-breaking)
# ===============================


def prepare_ranking_input(
    df: pd.DataFrame | None,
    label_date: Any | None,
    required_cols: list[str] | tuple[str, ...] | None = None,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare ranking input by validating columns and filtering by label_date.

    - Ensures required columns exist (records missing list in counts)
    - If label_date is provided, filters rows to that date (normalized)
    - Returns filtered df (possibly empty) and counts for diagnostics
    """
    counts: dict[str, Any] = {
        "rows_total": 0,
        "rows_for_label_date": 0,
        "missing_required": [],
    }

    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(), counts

    work = df.copy()
    counts["rows_total"] = int(len(work))

    # Required columns check
    try:
        if required_cols:
            missing = [c for c in required_cols if c not in work.columns]
            if missing:
                counts["missing_required"] = list(map(str, missing))
                # Return empty but keep counts for diagnostics visibility
                return pd.DataFrame(columns=list(work.columns)), counts
    except Exception:
        # Soft-fail: Do not raise; just proceed without required check
        pass

    # Label date filtering (normalize to Timestamp/normalize())
    if label_date is not None:
        try:
            lbl = pd.Timestamp(str(label_date)).normalize()
            if date_col in work.columns:
                work[date_col] = pd.to_datetime(
                    work[date_col], errors="coerce"
                ).dt.normalize()
                work = work.loc[work[date_col] == lbl]
            else:
                # try using index as date if DataFrame is time-indexed
                if isinstance(work.index, pd.DatetimeIndex):
                    work = work.loc[work.index.normalize() == lbl]
        except Exception:
            # If date parse fails, return empty selection but keep total count
            work = work.iloc[0:0]

    counts["rows_for_label_date"] = int(len(work))
    return work, counts


def apply_thresholds(
    df: pd.DataFrame,
    rules: dict[str, dict[str, Any]] | None,
    *,
    symbol_col: str = "symbol",
) -> tuple[pd.DataFrame, dict[str, int], dict[str, list[str]]]:
    """Apply threshold rules to a DataFrame and report exclude breakdown.

    rules format example:
        {
          "drop3d": {"op": ">=", "value": 0.125},
          "atr_ratio": {"op": ">=", "value": 0.05}
        }
    Returns: (filtered_df, reason_counts, reason_symbols)
    """
    if df is None or df.empty or not rules:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(), {}, {}

    work = df.copy()
    reason_counts: dict[str, int] = {}
    reason_symbols: dict[str, list[str]] = {}

    def _cmp(series: pd.Series, op: str, val: Any) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if op == ">=":
            return s >= val
        if op == ">":
            return s > val
        if op == "<=":
            return s <= val
        if op == "<":
            return s < val
        if op == "==":
            return s == val
        if op == "!=":
            return s != val
        # default: treat as invalid operator -> no row passes
        return pd.Series(False, index=s.index)

    # start from all-True and AND each rule
    keep_mask = pd.Series(True, index=work.index)
    for col, spec in rules.items():
        try:
            op = str(spec.get("op", ">=")).strip()
            val = spec.get("value", None)
            cond = _cmp(work[col], op, val)
        except Exception:
            # missing column -> reject all for this rule
            cond = pd.Series(False, index=work.index)

        # rows failing this rule
        fail_mask = (~cond) & keep_mask
        if fail_mask.any():
            reason_counts[col] = int(fail_mask.sum())
            if symbol_col in work.columns:
                try:
                    reason_symbols[col] = (
                        work.loc[fail_mask, symbol_col].astype(str).head(50).tolist()
                    )
                except Exception:
                    reason_symbols[col] = []
            else:
                reason_symbols[col] = []

        keep_mask &= cond

    # finally filter
    return work.loc[keep_mask].copy(), reason_counts, reason_symbols


def finalize_ranking_and_diagnostics(
    diagnostics: dict[str, Any],
    ranked_df: pd.DataFrame | None,
    *,
    ranking_source: str,
    extras: dict[str, Any] | None = None,
) -> None:
    """Finalize diagnostics after ranking and attach optional extras.

    - Sets ranked_top_n_count, final_top_n_count, ranking_source
    - Keeps setup_predicate_count as-is (caller sets beforehand)
    - Merges extras (keys not in standard safe diag become diagnostics_extra at export)
    """
    set_diagnostics_after_ranking(
        diagnostics, final_df=ranked_df, ranking_source=ranking_source
    )
    # mirror to final_top_n_count for snapshot consumers
    try:
        diagnostics["final_top_n_count"] = int(diagnostics.get("ranked_top_n_count", 0))
    except Exception:
        diagnostics["final_top_n_count"] = diagnostics.get("ranked_top_n_count", 0)

    if extras and isinstance(extras, dict):
        try:
            diagnostics.update(extras)
        except Exception:
            pass


__all__ += [
    "prepare_ranking_input",
    "apply_thresholds",
    "finalize_ranking_and_diagnostics",
]
