"""Candidate frame normalization and validation helpers.

Central place to enforce the candidate DataFrame contract used by the
allocation stage. Provides normalization (canonical column names) and a
lightweight validator that returns diagnostics useful for monitoring.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _to_float_series(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        # Fallback: construct a series of NaNs of same length
        return pd.Series([float("nan")] * len(series), index=series.index)


def normalize_candidate_frame(df: pd.DataFrame | None, system_name: str | None = None) -> pd.DataFrame:
    """Return a copy of df with canonical candidate columns.

    Canonical columns:
      - symbol (STR, upper)
      - system (STR, lower)
      - Close (float)
      - atr10 (float)
      - entry_price (float, may be NaN)
      - stop_price (float, may be NaN)
      - side (str)

    The function is defensive and will not raise; it guarantees the returned
    frame contains the canonical columns (with NaNs where unknown).
    """
    if df is None:
        return pd.DataFrame()
    out = df.copy()

    # symbol
    if "symbol" in out.columns:
        try:
            out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
        except Exception:
            out["symbol"] = out["symbol"].astype(str)
    else:
        # try to find a candidate-like column
        possible = [c for c in out.columns if c.lower() == "symbol"]
        if possible:
            out["symbol"] = out[possible[0]].astype(str).str.strip().str.upper()
        else:
            out["symbol"] = ""

    # system
    if "system" in out.columns:
        try:
            out["system"] = out["system"].astype(str).str.strip().str.lower()
        except Exception:
            out["system"] = out["system"].astype(str)
    else:
        out["system"] = str(system_name or "").strip().lower()

    # Close canonicalization (prefer existing 'Close', fallback to 'close')
    if "Close" in out.columns:
        try:
            out["Close"] = _to_float_series(out["Close"])
        except Exception:
            out["Close"] = _to_float_series(pd.Series([float("nan")] * len(out)))
    elif "close" in out.columns:
        out["Close"] = _to_float_series(out["close"])
    else:
        # try any case-insensitive match
        found = next((c for c in out.columns if c.lower() == "close"), None)
        if found:
            out["Close"] = _to_float_series(out[found])
        else:
            out["Close"] = pd.Series([float("nan")] * len(out))

    # ATR canonicalization
    # Systems sometimes emit 'atr', 'atr10', 'ATR10', or 'atr20' etc.
    # Prefer explicit 'atr10' if present, then generic 'atr', then any
    # column that starts with 'atr'. This maximises compatibility with
    # downstream sizing code that expects 'atr10'.
    if "atr10" in out.columns:
        out["atr10"] = _to_float_series(out["atr10"])
    elif "ATR10" in out.columns:
        out["atr10"] = _to_float_series(out["ATR10"])
    elif "atr" in out.columns:
        out["atr10"] = _to_float_series(out["atr"])
    else:
        # pick first column that looks like an ATR variant (atr*, ATR*)
        found = next((c for c in out.columns if c.lower().startswith("atr")), None)
        if found:
            out["atr10"] = _to_float_series(out[found])
        else:
            out["atr10"] = pd.Series([float("nan")] * len(out))

    # entry_price / stop_price normalization
    if "entry_price" in out.columns:
        out["entry_price"] = _to_float_series(out["entry_price"])
    else:
        found = next(
            (c for c in out.columns if c.lower() in ("entry_price", "entryprice")),
            None,
        )
        if found:
            out["entry_price"] = _to_float_series(out[found])
        else:
            out["entry_price"] = pd.Series([float("nan")] * len(out))

    if "stop_price" in out.columns:
        out["stop_price"] = _to_float_series(out["stop_price"])
    else:
        found = next(
            (c for c in out.columns if c.lower() in ("stop_price", "stopprice")),
            None,
        )
        if found:
            out["stop_price"] = _to_float_series(out[found])
        else:
            out["stop_price"] = pd.Series([float("nan")] * len(out))

    # side
    if "side" not in out.columns:
        out["side"] = "long"
    else:
        try:
            out["side"] = out["side"].astype(str).str.strip().str.lower()
        except Exception:
            out["side"] = out["side"].astype(str)

    # Ensure canonical order for readability
    canonical = [
        "symbol",
        "system",
        "side",
        "entry_price",
        "stop_price",
        "Close",
        "atr10",
    ]
    for c in canonical:
        if c not in out.columns:
            out.insert(len(out.columns), c, pd.Series([float("nan")] * len(out)))

    return out


def validate_candidate_frame(df: pd.DataFrame | None, required_fields: List[str] | None = None) -> Dict[str, Any]:
    """Return simple diagnostics about the candidate frame validity.

    Does not raise; returns dict with counts useful for monitoring.
    """
    if df is None:
        return {
            "rows_total": 0,
            "missing_counts": {},
            "rows_missing_entry": 0,
            "rows_missing_price": 0,
        }

    out: Dict[str, Any] = {}
    rows = int(len(df))
    out["rows_total"] = rows

    # Build a conservative set of fields to report missing counts for so
    # diagnostics are stable across callers. We always report these keys
    # even if the incoming frame doesn't contain them.
    fields_to_report = ["symbol", "Close", "entry_price", "atr10"]
    missing: Dict[str, int] = {}
    for f in fields_to_report:
        if f in df.columns:
            try:
                missing[f] = int(df[f].isna().sum())
            except Exception:
                missing[f] = rows
        else:
            # If column does not exist treat as all-missing for monitoring
            missing[f] = rows

    out["missing_counts"] = missing

    # entry-specific: rows where entry_price is absent/NaN
    try:
        if "entry_price" in df.columns:
            try:
                out["rows_missing_entry"] = int(df["entry_price"].isna().sum())
            except Exception:
                out["rows_missing_entry"] = rows
        else:
            out["rows_missing_entry"] = rows
    except Exception:
        out["rows_missing_entry"] = rows

    # price-specific: rows that have neither an entry_price nor a Close
    try:
        has_close = "Close" in df.columns and not df["Close"].isna().all()
        has_entry = "entry_price" in df.columns and not df["entry_price"].isna().all()
        if not has_close and not has_entry:
            # All rows lack any usable price information
            out["rows_missing_price"] = rows
        else:
            # Count rows where both are missing
            try:
                if "Close" in df.columns:
                    cond_close = df["Close"].isna()
                else:
                    cond_close = pd.Series([True] * rows)

                if "entry_price" in df.columns:
                    cond_entry = df["entry_price"].isna()
                else:
                    cond_entry = pd.Series([True] * rows)

                out["rows_missing_price"] = int((cond_close & cond_entry).sum())
            except Exception:
                out["rows_missing_price"] = rows
    except Exception:
        out["rows_missing_price"] = rows

    return out
