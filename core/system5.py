"""System5 core logic (Long mean-reversion with high ADX).

High ADX mean-reversion strategy:
- Indicators: adx7, atr10, dollarvolume20, atr_pct (precomputed only)
- Setup conditions: Close>=5, AvgVol50>500k, DV50>2.5M, ATR_Pct>2.5%, Close>SMA100+ATR10, ADX7>55, RSI3<50
- Candidate generation: ADX7 descending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pandas as pd

from common.batch_processing import process_symbols_batch
from common.system_common import check_precomputed_indicators, get_total_days
from common.system_constants import SYSTEM5_REQUIRED_INDICATORS
from common.system_setup_predicates import validate_predicate_equivalence
from common.utils import get_cached_data

# ATR percentage threshold for System5 filtering
DEFAULT_ATR_PCT_THRESHOLD = 0.025


def format_atr_pct_threshold_label(threshold: float | None = None) -> str:
    """UI/ログ用のATR閾値ラベルを一元化。scripts/today や today_signals で利用。"""
    actual_threshold = threshold if threshold is not None else DEFAULT_ATR_PCT_THRESHOLD
    return f"> {actual_threshold:.2%}"


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Check precomputed indicators and apply System5-specific filters.

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
        missing_indicators = [
            col for col in SYSTEM5_REQUIRED_INDICATORS if col not in df.columns
        ]
        if missing_indicators:
            return symbol, None

        # Apply System5-specific filters and setup
        x = df.copy()

        # Filter: Close>=5, ADX7>35, ATR_Pct>2.5% (high volatility trend)
        x["filter"] = (
            (x["Close"] >= 5.0)
            & (x["adx7"] > 35.0)
            & (x["atr_pct"] > DEFAULT_ATR_PCT_THRESHOLD)
        )

        # Setup: Same as filter for System5 (simple high ADX trend selection)
        x["setup"] = x["filter"]

        return symbol, x

    except Exception:
        return symbol, None


def prepare_data_vectorized_system5(
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
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """System5 data preparation processing (high ADX mean-reversion strategy).

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
            valid_data_dict, error_symbols = check_precomputed_indicators(
                raw_data_dict, SYSTEM5_REQUIRED_INDICATORS, "System5", skip_callback
            )

            if valid_data_dict:
                # Apply System5-specific filters
                prepared_dict = {}
                for symbol, df in valid_data_dict.items():
                    x = df.copy()

                    # Filter: Close>=5, ADX7>35, ATR_Pct>2.5% (high volatility trend)
                    x["filter"] = (
                        (x["Close"] >= 5.0)
                        & (x["adx7"] > 35.0)
                        & (x["atr_pct"] > DEFAULT_ATR_PCT_THRESHOLD)
                    )

                    # Setup: Same as filter for System5 (simple high ADX trend selection)
                    x["setup"] = x["filter"]

                    prepared_dict[symbol] = x

                if log_callback:
                    log_callback(
                        f"System5: Fast-path processed {len(prepared_dict)} symbols"
                    )

                return prepared_dict

        except RuntimeError:
            # Re-raise error immediately if required indicators are missing
            raise
        except Exception:
            # Fall back to normal processing for other errors
            if log_callback:
                log_callback(
                    "System5: Fast-path failed, falling back to normal processing"
                )

    # Normal processing path: batch processing from symbol list
    if symbols:
        target_symbols = symbols
    elif raw_data_dict:
        target_symbols = list(raw_data_dict.keys())
    else:
        if log_callback:
            log_callback("System5: No symbols provided, returning empty dict")
        return {}

    if log_callback:
        log_callback(
            f"System5: Starting normal processing for {len(target_symbols)} symbols"
        )

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
        system_name="System5",
    )
    try:
        validate_predicate_equivalence(results, "5", log_fn=log_callback)
    except Exception:
        pass
    return results


def generate_candidates_system5(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    batch_size: int | None = None,
    latest_only: bool = False,
    include_diagnostics: bool = False,
    diagnostics: dict[str, Any] | None = None,
    **kwargs: Any,
) -> (
    tuple[dict[pd.Timestamp, dict[str, dict]], pd.DataFrame | None]
    | tuple[dict[pd.Timestamp, dict[str, dict]], pd.DataFrame | None, dict[str, Any]]
):
    """System5 candidate generation (ADX7 descending ranking).

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
            log_callback("System5: No data provided for candidate generation")
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
                # 'setup' 列が未生成ならスキップせず、存在する場合のみ False を除外
                setup_col_val = (
                    bool(last_row.get("setup", False)) if "setup" in last_row else True
                )
                from common.system_setup_predicates import (
                    system5_setup_predicate as _s5_pred,
                )

                pred_val = _s5_pred(last_row)
                if pred_val:
                    diagnostics["setup_predicate_count"] += 1
                if pred_val and not setup_col_val:
                    diagnostics["predicate_only_pass_count"] += 1
                    diagnostics["mismatch_flag"] = 1
                if not setup_col_val:
                    continue
                adx7_val = last_row.get("adx7", 0)
                try:
                    if pd.isna(adx7_val) or float(adx7_val) <= 35.0:
                        continue
                except Exception:
                    continue
                dt = df.index[-1]
                date_counter[dt] = date_counter.get(dt, 0) + 1
                rows.append(
                    {
                        "symbol": sym,
                        "date": dt,
                        "adx7": adx7_val,
                        "atr_pct": last_row.get("atr_pct", 0),
                        "close": last_row.get("Close", 0),
                    }
                )
            if not rows:
                if log_callback:
                    log_callback("System5: latest_only fast-path produced 0 rows")
                return ({}, None, diagnostics) if include_diagnostics else ({}, None)
            df_all = pd.DataFrame(rows)
            try:
                mode_date = max(date_counter.items(), key=lambda kv: kv[1])[0]
                df_all = df_all[df_all["date"] == mode_date]
            except Exception:
                pass
            df_all = df_all.sort_values("adx7", ascending=False, kind="stable").head(
                top_n
            )
            diagnostics["final_top_n_count"] = len(df_all)
            diagnostics["ranking_source"] = "latest_only"
            by_date: dict[pd.Timestamp, dict[str, dict]] = {}
            for dt_raw, sub in df_all.groupby("date"):
                dt = pd.Timestamp(str(dt_raw))
                symbol_map: dict[str, dict[str, Any]] = {}
                for rec in sub.to_dict("records"):
                    sym_val = rec.get("symbol")
                    if not isinstance(sym_val, str) or not sym_val:
                        continue
                    payload: dict[str, Any] = {
                        str(k): v for k, v in rec.items() if k not in ("symbol", "date")
                    }
                    symbol_map[str(sym_val)] = payload
                by_date[dt] = symbol_map
            if log_callback:
                log_callback(
                    f"System5: latest_only fast-path -> {len(df_all)} candidates (symbols={len(rows)})"
                )
            return (
                (by_date, df_all.copy(), diagnostics)
                if include_diagnostics
                else (by_date, df_all.copy())
            )
        except Exception as e:
            if log_callback:
                log_callback(f"System5: fast-path failed -> fallback ({e})")
            pass

    # Aggregate all dates
    all_dates_set: set[pd.Timestamp] = set()
    for df in prepared_dict.values():
        if df is not None and not df.empty:
            all_dates_set.update(df.index)

    if not all_dates_set:
        if log_callback:
            log_callback("System5: No valid dates found in data")
        return ({}, None, diagnostics) if include_diagnostics else ({}, None)
    all_dates = sorted(all_dates_set)

    candidates_by_date: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    all_candidates: list[dict[str, Any]] = []

    if log_callback:
        log_callback(f"System5: Generating candidates for {len(all_dates)} dates")

    # Execute ADX7 ranking by date (descending - highest ADX7 first)
    for i, date in enumerate(all_dates):
        date_candidates = []

        for symbol, df in prepared_dict.items():
            try:
                if df is None or date not in df.index:
                    continue
                row = cast(pd.Series, df.loc[date])
                setup_val = bool(row.get("setup", False))
                from common.system_setup_predicates import (
                    system5_setup_predicate as _s5_pred,
                )

                pred_val = _s5_pred(row)
                if pred_val:
                    diagnostics["setup_predicate_count"] += 1
                if pred_val and not setup_val:
                    diagnostics["predicate_only_pass_count"] += 1
                    diagnostics["mismatch_flag"] = 1
                if not bool(setup_val):
                    continue
                adx7_val = cast(Any, row.get("adx7", 0))
                try:
                    if pd.isna(adx7_val) or float(adx7_val) <= 35.0:
                        continue
                except Exception:
                    continue

                date_candidates.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "adx7": adx7_val,
                        "atr_pct": row.get("atr_pct", 0),
                        "close": row.get("Close", 0),
                    }
                )

            except Exception:
                continue

        # Sort by ADX7 descending (highest first) and extract top_n
        if date_candidates:
            date_candidates.sort(key=lambda x: x["adx7"], reverse=True)
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
        candidates_df = candidates_df.sort_values(
            ["date", "adx7"], ascending=[True, False]
        )
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
            f"System5: Generated {total_candidates} candidates across {unique_dates} dates"
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


def get_total_days_system5(data_dict: dict[str, pd.DataFrame]) -> int:
    """Get total days count for System5 data.

    Args:
        data_dict: Data dictionary

    Returns:
        Maximum day count
    """
    return get_total_days(data_dict)


__all__ = [
    "prepare_data_vectorized_system5",
    "generate_candidates_system5",
    "get_total_days_system5",
]
