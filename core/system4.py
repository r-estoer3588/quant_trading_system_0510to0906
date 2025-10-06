"""System4 core logic (Long trend low-vol pullback).

Trend low-volatility pullback strategy:
- Indicators: rsi4, sma200, atr40, hv50, dollarvolume50 (precomputed only)
- Filter conditions: DollarVolume50>100M, HV50 10-40% (volatility contraction)
- Setup conditions: Filter + Close>SMA200 (trend confirmation)
- Candidate generation: RSI4 ascending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pandas as pd

from common.batch_processing import process_symbols_batch
from common.system_common import check_precomputed_indicators, get_total_days
from common.system_constants import SYSTEM4_REQUIRED_INDICATORS
from common.system_setup_predicates import validate_predicate_equivalence
from common.utils import get_cached_data


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Check precomputed indicators and apply System4-specific filters.

    Args:
        symbol: Target symbol to process

    Returns:
        (symbol, processed DataFrame | None)
    """
    try:
        df = get_cached_data(symbol)
        if df is None or df.empty:
            return symbol, None

        # Check for required indicators
        missing_indicators = [col for col in SYSTEM4_REQUIRED_INDICATORS if col not in df.columns]
        if missing_indicators:
            return symbol, None

        # Apply System4-specific filters and setup
        x = df.copy()

        # Filter: DollarVolume50>100M, HV50 10-40% (volatility contraction)
        x["filter"] = (x["dollarvolume50"] > 100_000_000) & x["hv50"].between(10, 40)

        # Setup: Filter + Close>SMA200 (trend confirmation)
        x["setup"] = x["filter"] & (x["Close"] > x["sma200"])

        return symbol, x

    except Exception:
        return symbol, None


def prepare_data_vectorized_system4(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    *,
    progress_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    skip_callback: Callable[[str, str], None] | None = None,
    batch_size: int | None = None,
    reuse_indicators: bool = True,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    **_unused_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """System4 data preparation processing (trend low-volatility pullback strategy).

    Execute high-speed processing using precomputed indicators.

    Args:
        raw_data_dict: Raw data dictionary (None to fetch from cache)
        progress_callback: Progress reporting callback
        log_callback: Log output callback
        skip_callback: Error skip callback
        batch_size: Batch size
        reuse_indicators: Reuse existing indicators (for speed)
        symbols: Target symbol list
        use_process_pool: Process pool usage flag
        max_workers: Maximum worker count

    Returns:
        Processed data dictionary
    """
    # Fast path: reuse precomputed indicators
    if reuse_indicators and raw_data_dict:
        try:
            # Early check - verify required indicators exist
            valid_data_dict, _error_symbols = check_precomputed_indicators(
                raw_data_dict, SYSTEM4_REQUIRED_INDICATORS, "System4", skip_callback
            )

            if valid_data_dict:
                # Apply System4-specific filters
                prepared_dict = {}
                for symbol, df in valid_data_dict.items():
                    x = df.copy()

                    # Filter: DollarVolume50>100M, HV50 10-40% (volatility contraction)
                    x["filter"] = (x["dollarvolume50"] > 100_000_000) & x["hv50"].between(10, 40)

                    # Setup: Filter + Close>SMA200 (trend confirmation)
                    x["setup"] = x["filter"] & (x["Close"] > x["sma200"])

                    prepared_dict[symbol] = x

                if log_callback:
                    log_callback(f"System4: Fast-path processed {len(prepared_dict)} symbols")

                return prepared_dict

        except RuntimeError:
            # Re-raise error immediately if required indicators are missing
            raise
        except Exception:
            # Fall back to normal processing for other errors
            if log_callback:
                log_callback("System4: Fast-path failed, falling back to normal processing")

    # Normal processing path: batch processing from symbol list
    if symbols:
        target_symbols = symbols
    elif raw_data_dict:
        target_symbols = list(raw_data_dict.keys())
    else:
        if log_callback:
            log_callback("System4: No symbols provided, returning empty dict")
        return {}

    if log_callback:
        log_callback(f"System4: Starting normal processing for {len(target_symbols)} symbols")

    # Execute batch processing
    results, error_symbols = process_symbols_batch(
        target_symbols,
        _compute_indicators,
        batch_size=batch_size,
        use_process_pool=use_process_pool,
        max_workers=max_workers,
        progress_callback=progress_callback,
        log_callback=log_callback,
        skip_callback=skip_callback,
        system_name="System4",
    )
    try:
        validate_predicate_equivalence(results, "4", log_fn=log_callback)
    except Exception:
        pass
    return results


def generate_candidates_system4(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    latest_only: bool = False,
    include_diagnostics: bool = False,
    diagnostics: dict[str, Any] | None = None,
    **_unused_kwargs: Any,
) -> (
    tuple[dict[pd.Timestamp, dict[str, dict[str, Any]]], pd.DataFrame | None]
    | tuple[
        dict[pd.Timestamp, dict[str, dict[str, Any]]],
        pd.DataFrame | None,
        dict[str, Any],
    ]
):
    """System4 candidate generation (RSI4 ascending ranking).

    Args:
        prepared_dict: Prepared data dictionary
        top_n: Number of top entries to extract
        progress_callback: Progress reporting callback
        log_callback: Log output callback

    Returns:
        (Daily candidate dictionary, Integrated candidate DataFrame)
    """
    if diagnostics is None:
        diagnostics = {
            "ranking_source": None,
            "setup_predicate_count": 0,
            "final_top_n_count": 0,
            "predicate_only_pass_count": 0,
            "mismatch_flag": 0,
        }

    if not prepared_dict:
        if log_callback:
            log_callback("System4: No data provided for candidate generation")
        return ({}, None, diagnostics) if include_diagnostics else ({}, None)

    if top_n is None:
        top_n = 20  # Default value

    if latest_only:
        try:
            rows: list[dict] = []
            date_counter: dict[pd.Timestamp, int] = {}
            for sym, df in prepared_dict.items():
                if df is None or df.empty:
                    continue
                last_row = df.iloc[-1]
                # setup==True のみ採用（列が無ければ不採用）
                if not bool(last_row.get("setup", False)):
                    continue
                rsi4_val = last_row.get("rsi4", None)
                try:
                    if rsi4_val is None or pd.isna(rsi4_val):
                        continue
                except Exception:
                    continue
                dt = pd.Timestamp(df.index[-1])
                date_counter[dt] = date_counter.get(dt, 0) + 1
                rows.append(
                    {
                        "symbol": sym,
                        "date": dt,
                        "rsi4": rsi4_val,
                        "atr_ratio": last_row.get("atr_ratio", 0),
                        "close": last_row.get("Close", 0),
                        "sma200": last_row.get("sma200", 0),
                    }
                )
            if not rows:
                if log_callback:
                    try:
                        samples: list[str] = []
                        taken = 0
                        for s_sym, s_df in prepared_dict.items():
                            if s_df is None or getattr(s_df, "empty", True):
                                continue
                            try:
                                s_last = s_df.iloc[-1]
                                s_dt = pd.to_datetime(str(s_df.index[-1])).normalize()
                                s_setup = bool(s_last.get("setup", False))
                                s_rsi = s_last.get("rsi4", float("nan"))
                                try:
                                    s_rsi_f = float(s_rsi)
                                except Exception:
                                    s_rsi_f = float("nan")
                                samples.append(
                                    (
                                        f"{s_sym}: date={s_dt.date()} setup={s_setup} "
                                        f"rsi4={s_rsi_f:.4f}"
                                    )
                                )
                                taken += 1
                                if taken >= 2:
                                    break
                            except Exception:
                                continue
                        if samples:
                            log_callback(
                                ("System4: DEBUG latest_only 0 candidates. " + " | ".join(samples))
                            )
                    except Exception:
                        pass
                    log_callback("System4: latest_only fast-path produced 0 rows")
                return ({}, None, diagnostics) if include_diagnostics else ({}, None)
            df_all = pd.DataFrame(rows)
            try:
                mode_date = max(date_counter.items(), key=lambda kv: kv[1])[0]
                df_all = df_all[df_all["date"] == mode_date]
            except Exception:
                pass
            df_all = df_all.sort_values("rsi4", ascending=True, kind="stable").head(top_n)
            diagnostics["final_top_n_count"] = len(df_all)
            diagnostics["ranking_source"] = "latest_only"
            by_date: dict[pd.Timestamp, dict[str, dict]] = {}
            for dt_raw, sub in df_all.groupby("date"):
                # safe cast for mypy (numpy scalar -> str)
                dt = pd.Timestamp(str(dt_raw))
                symbol_map: dict[str, dict[str, Any]] = {}
                for rec in sub.to_dict("records"):
                    sym = rec.get("symbol")
                    if not sym:
                        continue
                    payload: dict[str, Any] = {
                        str(k): v for k, v in rec.items() if k not in ("symbol", "date")
                    }
                    symbol_map[str(sym)] = payload
                by_date[dt] = symbol_map
            if log_callback:
                log_callback(
                    (
                        "System4: latest_only fast-path -> "
                        f"{len(df_all)} candidates (symbols={len(rows)})"
                    )
                )
            return (
                (by_date, df_all.copy(), diagnostics)
                if include_diagnostics
                else (by_date, df_all.copy())
            )
        except Exception as e:
            if log_callback:
                log_callback(f"System4: fast-path failed -> fallback ({e})")
            # fall back to normal path below

    # Aggregate all dates
    all_dates_set: set[pd.Timestamp] = set()
    for df in prepared_dict.values():
        if df is not None and not df.empty:
            all_dates_set.update(df.index)

    if not all_dates_set:
        if log_callback:
            log_callback("System4: No valid dates found in data")
        return ({}, None, diagnostics) if include_diagnostics else ({}, None)
    all_dates = sorted(all_dates_set)

    candidates_by_date: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    all_candidates: list[dict[str, Any]] = []

    if log_callback:
        log_callback(f"System4: Generating candidates for {len(all_dates)} dates")

    # Execute RSI4 ranking by date (ascending - lowest RSI4 first for oversold)
    for i, date in enumerate(all_dates):
        date_candidates = []

        for symbol, df in prepared_dict.items():
            try:
                if df is None or date not in df.index:
                    continue
                row = cast(pd.Series, df.loc[date])
                setup_val = bool(row.get("setup", False))
                from common.system_setup_predicates import system4_setup_predicate as _s4_pred

                pred_val = _s4_pred(row)
                if pred_val:
                    diagnostics["setup_predicate_count"] += 1
                if pred_val and not setup_val:
                    diagnostics["predicate_only_pass_count"] += 1
                    diagnostics["mismatch_flag"] = 1
                if not bool(setup_val):
                    continue
                rsi4_val = cast(Any, row.get("rsi4", 100))
                try:
                    if pd.isna(rsi4_val) or float(rsi4_val) >= 30.0:
                        continue
                except Exception:
                    continue

                date_candidates.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "rsi4": rsi4_val,
                        "atr_ratio": row.get("atr_ratio", 0),
                        "close": row.get("Close", 0),
                        "sma200": row.get("sma200", 0),
                    }
                )

            except Exception:
                continue

        # Sort by RSI4 ascending (lowest first) and extract top_n
        if date_candidates:
            date_candidates.sort(key=lambda x: x["rsi4"])
            top_candidates = date_candidates[:top_n]

            candidates_by_date[date] = top_candidates
            all_candidates.extend(top_candidates)

        # Progress reporting
        if progress_callback and (i + 1) % max(1, len(all_dates) // 10) == 0:
            progress_callback(f"Processed {i + 1}/{len(all_dates)} dates")

    # Create integrated DataFrame
    if all_candidates:
        candidates_df = pd.DataFrame(all_candidates)
        candidates_df["date"] = pd.to_datetime(candidates_df["date"])
        candidates_df = candidates_df.sort_values(["date", "rsi4"], ascending=[True, True])
        diagnostics["ranking_source"] = "full_scan"
        try:
            last_dt = max(candidates_by_date.keys())
            diagnostics["final_top_n_count"] = len(candidates_by_date.get(last_dt, []))
        except Exception:
            diagnostics["final_top_n_count"] = 0
    else:
        candidates_df = None

    if log_callback:
        total_candidates = len(all_candidates)
        unique_dates = len(candidates_by_date)
        log_callback(
            ("System4: Generated " f"{total_candidates} candidates across {unique_dates} dates")
        )

    normalized: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    for dt, recs in candidates_by_date.items():
        out_symbol_map: dict[str, dict[str, Any]] = {}
        for rec in recs:
            sym_any = rec.get("symbol")
            if not isinstance(sym_any, str) or not sym_any:
                continue
            payload = {k: v for k, v in rec.items() if k not in ("symbol", "date")}
            out_symbol_map[sym_any] = payload
        normalized[dt] = out_symbol_map
    return (
        (normalized, candidates_df, diagnostics)
        if include_diagnostics
        else (normalized, candidates_df)
    )


def get_total_days_system4(data_dict: dict[str, pd.DataFrame]) -> int:
    """Get total days count for System4 data.

    Args:
        data_dict: Data dictionary

    Returns:
        Maximum day count
    """
    return get_total_days(data_dict)


__all__ = [
    "prepare_data_vectorized_system4",
    "generate_candidates_system4",
    "get_total_days_system4",
]
