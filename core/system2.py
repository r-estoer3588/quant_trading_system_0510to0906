"""System2 core logic (Short RSI spike).

RSI3-based short spike strategy:
- Indicators: rsi3, adx7, atr10, dollarvolume20, atr_ratio, twodayup (precomputed only)
- Setup conditions: Close>5, DollarVolume20>25M, ATR_Ratio>0.03, RSI3>90, twodayup
- Candidate generation: ADX7 descending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pandas as pd

from common.batch_processing import process_symbols_batch
from common.system_common import check_precomputed_indicators, get_total_days
from common.system_constants import SYSTEM2_REQUIRED_INDICATORS
from common.system_setup_predicates import validate_predicate_equivalence
from common.utils import get_cached_data


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Check precomputed indicators and apply System2-specific filters.

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
        missing_indicators = [col for col in SYSTEM2_REQUIRED_INDICATORS if col not in df.columns]
        if missing_indicators:
            return symbol, None

        # Apply System2-specific filters and setup
        x = df.copy()

        # Filter: Close>=5, DollarVolume20>25M, ATR_Ratio>0.03
        x["filter"] = (
            (x["Close"] >= 5.0) & (x["dollarvolume20"] > 25_000_000) & (x["atr_ratio"] > 0.03)
        )

        # Setup: Filter + RSI3>90 + twodayup
        x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]

        return symbol, x

    except Exception:
        return symbol, None


def prepare_data_vectorized_system2(
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
    """System2 data preparation processing (RSI3 spike strategy).

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
                raw_data_dict, SYSTEM2_REQUIRED_INDICATORS, "System2", skip_callback
            )

            if valid_data_dict:
                # Apply System2-specific filters
                prepared_dict = {}
                for symbol, df in valid_data_dict.items():
                    x = df.copy()

                    # Filter: Close>=5, DollarVolume20>25M, ATR_Ratio>0.03
                    x["filter"] = (
                        (x["Close"] >= 5.0)
                        & (x["dollarvolume20"] > 25_000_000)
                        & (x["atr_ratio"] > 0.03)
                    )

                    # Setup: Filter + RSI3>90 + twodayup
                    x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]

                    prepared_dict[symbol] = x

                if log_callback:
                    log_callback(f"System2: Fast-path processed {len(prepared_dict)} symbols")

                return prepared_dict

        except RuntimeError:
            # Re-raise error immediately if required indicators are missing
            raise
        except Exception:
            # Fall back to normal processing for other errors
            if log_callback:
                log_callback("System2: Fast-path failed, falling back to normal processing")

    # Normal processing path: batch processing from symbol list
    if symbols:
        target_symbols = symbols
    elif raw_data_dict:
        target_symbols = list(raw_data_dict.keys())
    else:
        if log_callback:
            log_callback("System2: No symbols provided, returning empty dict")
        return {}

    if log_callback:
        log_callback(f"System2: Starting normal processing for {len(target_symbols)} symbols")

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
        system_name="System2",
    )
    try:
        validate_predicate_equivalence(results, "2", log_fn=log_callback)
    except Exception:
        pass
    return results


def generate_candidates_system2(
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
    """System2 candidate generation (ADX7 descending ranking).

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
            # single-source-of-truth: ranked_top_n_count only
            "ranked_top_n_count": 0,
            "predicate_only_pass_count": 0,
            "mismatch_flag": 0,
        }

    if not prepared_dict:
        if log_callback:
            log_callback("System2: No data provided for candidate generation")
        return ({}, None, diagnostics) if include_diagnostics else ({}, None)

    if top_n is None:
        top_n = 20  # Default value

    # === Fast Path (latest_only) ===
    # 当日シグナル抽出用途: 最新日のみを対象に O(S) でランキング
    if latest_only:
        try:
            rows: list[dict] = []
            date_counter: dict[pd.Timestamp, int] = {}
            setup_pass_count = 0  # カウンター追加
            for sym, df in prepared_dict.items():
                if df is None or df.empty:
                    continue
                last_row = df.iloc[-1]

                # Use predicate-based evaluation (no setup column dependency)
                try:
                    from common.system_setup_predicates import system2_setup_predicate as _s2_pred
                except Exception:
                    _s2_pred = None

                setup_ok = False
                if _s2_pred is not None:
                    try:
                        setup_ok = bool(_s2_pred(last_row))
                    except Exception:
                        setup_ok = False

                if not setup_ok:
                    continue

                setup_pass_count += 1  # setup通過カウント

                adx7_val = last_row.get("adx7", None)
                try:
                    if adx7_val is None or pd.isna(adx7_val):
                        continue
                except Exception:
                    continue
                dt = pd.Timestamp(df.index[-1])
                date_counter[dt] = date_counter.get(dt, 0) + 1
                rows.append(
                    {
                        "symbol": sym,
                        "date": dt,
                        "adx7": adx7_val,
                        "rsi3": last_row.get("rsi3", 0),
                        "close": last_row.get("Close", 0),
                    }
                )

            diagnostics["setup_predicate_count"] = setup_pass_count  # 記録

            if not rows:
                if log_callback:
                    log_callback("System2: latest_only fast-path produced 0 rows")
                return ({}, None, diagnostics) if include_diagnostics else ({}, None)
            df_all = pd.DataFrame(rows)
            # 最頻日で揃える（欠落シンボル耐性）
            try:
                mode_date = max(date_counter.items(), key=lambda kv: kv[1])[0]
                df_all = df_all[df_all["date"] == mode_date]
            except Exception:
                pass
            df_all = df_all.sort_values("adx7", ascending=False, kind="stable").head(top_n)
            diagnostics["ranked_top_n_count"] = len(df_all)
            diagnostics["ranking_source"] = "latest_only"
            # 候補0件なら代表サンプルを1-2件だけDEBUGログ出力
            if diagnostics.get("ranked_top_n_count", 0) == 0 and log_callback:
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
                            s_adx = s_last.get("adx7", float("nan"))
                            samples.append(
                                (
                                    f"{s_sym}: date={s_dt.date()} "
                                    f"setup={s_setup} adx7={float(s_adx):.4f}"
                                )
                            )
                            taken += 1
                            if taken >= 2:
                                break
                        except Exception:
                            continue
                    if samples:
                        log_callback(
                            "System2: DEBUG latest_only 0 candidates. " + " | ".join(samples)
                        )
                except Exception:
                    pass
            # Orchestrator expects: {date: {symbol: {field: value}}}
            by_date: dict[pd.Timestamp, dict[str, dict]] = {}
            for dt_raw, sub in df_all.groupby("date"):
                # 明示的に文字列→Timestamp へ (型推論安定化)
                dt = pd.to_datetime(str(dt_raw))
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
                        "System2: latest_only fast-path -> "
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
                log_callback(f"System2: fast-path failed -> fallback ({e})")
            # フォールバックして従来ロジックへ続行
            pass

    # Aggregate all dates
    all_dates_set: set[pd.Timestamp] = set()
    for df in prepared_dict.values():
        if df is not None and not df.empty:
            all_dates_set.update(df.index)

    if not all_dates_set:
        if log_callback:
            log_callback("System2: No valid dates found in data")
        return ({}, None, diagnostics) if include_diagnostics else ({}, None)
    all_dates = sorted(all_dates_set)

    candidates_by_date: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    all_candidates: list[dict[str, Any]] = []

    if log_callback:
        log_callback(f"System2: Generating candidates for {len(all_dates)} dates")

    # Execute ADX7 ranking by date
    for i, date in enumerate(all_dates):
        date_candidates = []

        for symbol, df in prepared_dict.items():
            try:
                if df is None or date not in df.index:
                    continue
                row = cast(pd.Series, df.loc[date])
                setup_val = bool(row.get("setup", False))
                from common.system_setup_predicates import system2_setup_predicate as _s2_pred

                pred_val = _s2_pred(row)
                if pred_val:
                    diagnostics["setup_predicate_count"] += 1
                if pred_val and not setup_val:
                    diagnostics["predicate_only_pass_count"] += 1
                    diagnostics["mismatch_flag"] = 1
                if not setup_val:
                    continue
                adx7_val = cast(Any, row.get("adx7", 0))
                try:
                    if pd.isna(adx7_val) or float(adx7_val) <= 0:
                        continue
                except Exception:
                    continue

                date_candidates.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "adx7": adx7_val,
                        "rsi3": row.get("rsi3", 0),
                        "close": row.get("Close", 0),
                    }
                )

            except Exception:
                continue

        # Sort by ADX7 descending and extract top_n
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
        candidates_df = candidates_df.sort_values(["date", "adx7"], ascending=[True, False])
        last_date = max(candidates_by_date.keys()) if candidates_by_date else None
        if last_date is not None:
            diagnostics["ranked_top_n_count"] = len(candidates_by_date.get(last_date, []))
        diagnostics["ranking_source"] = "full_scan"
    else:
        candidates_df = None

    if log_callback:
        total_candidates = len(all_candidates)
        unique_dates = len(candidates_by_date)
        log_callback(
            ("System2: Generated " f"{total_candidates} candidates across {unique_dates} dates")
        )

    # Normalize to {date: {symbol: payload}}
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


def get_total_days_system2(data_dict: dict[str, pd.DataFrame]) -> int:
    """Get total days count for System2 data.

    Args:
        data_dict: Data dictionary

    Returns:
        Maximum day count
    """
    return get_total_days(data_dict)


__all__ = [
    "prepare_data_vectorized_system2",
    "generate_candidates_system2",
    "get_total_days_system2",
]
