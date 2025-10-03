"""System1 core logic (Long ROC200 momentum).

ROC200-based momentum strategy:
- Indicators: ROC200, SMA200, DollarVolume20 (precomputed only)
- Setup conditions: Close>5, DollarVolume20>25M, Close>SMA200, ROC200>0
- Candidate generation: ROC200 descending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only
"""

from __future__ import annotations

import os

import pandas as pd

from common.batch_processing import process_symbols_batch
from common.system_common import check_precomputed_indicators, get_total_days
from common.system_constants import SYSTEM1_REQUIRED_INDICATORS
from common.utils import get_cached_data

# --- Backward compatibility helpers for legacy direct tests ---
# Some tests (tests/test_system1_direct.py) expect internal helper functions
# that existed in the previous refactored version (see system1_backup.py).
# We reintroduce lightweight versions here without altering the new fast-path
# design. These are intentionally minimal and rely only on current imports.

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _rename_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy(deep=False)
    rename_map = {}
    for low, up in (
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ):
        if low in x.columns and up not in x.columns:
            rename_map[low] = up
    if rename_map:
        try:
            x = x.rename(columns=rename_map)
        except Exception:
            pass
    return x


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        idx = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    else:
        idx = pd.to_datetime(df.index, errors="coerce").normalize()
    x = df.copy(deep=False)
    x.index = pd.Index(idx, name="Date")
    x = x[~x.index.isna()]
    try:
        x = x.sort_index()
        if getattr(x.index, "has_duplicates", False):
            x = x[~x.index.duplicated(keep="last")]
    except Exception:
        pass
    return x


def _prepare_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("empty_frame")
    x = _rename_ohlcv(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in x.columns]
    if missing:
        raise ValueError(f"missing_cols:{','.join(missing)}")
    x = _normalize_index(x)
    for col in REQUIRED_COLUMNS:
        if col in x.columns:
            try:
                x[col] = pd.to_numeric(x[col], errors="coerce")
            except Exception:
                pass
    x = x.dropna(subset=[c for c in ("High", "Low", "Close") if c in x.columns])
    return x


def _compute_indicators_frame(df: pd.DataFrame) -> pd.DataFrame:
    # System1 now relies exclusively on precomputed indicators (fast path).
    x = df.copy(deep=False)
    required_indicators = ["sma25", "sma50", "roc200", "atr20", "dollarvolume20"]
    missing_indicators = [c for c in required_indicators if c not in x.columns]
    if missing_indicators:
        raise ValueError(f"missing precomputed indicators: {missing_indicators}")
    # Derive filter/setup (legacy style) for tests expecting them here.
    try:
        if "filter" not in x.columns:
            x["filter"] = (x["Low"] >= 5) & (x["dollarvolume20"] > 50_000_000)
        if "setup" not in x.columns:
            x["setup"] = x["filter"] & (x["sma25"] > x["sma50"])
    except Exception:
        # Fallback safe defaults
        x["filter"] = False
        x["setup"] = False
    return x


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Check precomputed indicators and apply System1-specific filters.

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
        missing_indicators = [col for col in SYSTEM1_REQUIRED_INDICATORS if col not in df.columns]
        if missing_indicators:
            return symbol, None

        # Apply System1-specific filters and setup
        x = df.copy()

        # Filter: Close>=5, DollarVolume20>25M
        x["filter"] = (x["Close"] >= 5.0) & (x["dollarvolume20"] > 25_000_000)

        # Setup: Filter + Close>SMA200 + ROC200>0
        x["setup"] = x["filter"] & (x["Close"] > x["sma200"]) & (x["roc200"] > 0)

        return symbol, x

    except Exception:
        return symbol, None


def prepare_data_vectorized_system1(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    *,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
    batch_size: int | None = None,
    reuse_indicators: bool = True,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    **_kwargs,
) -> dict[str, pd.DataFrame]:
    """System1 data preparation processing (ROC200 momentum strategy).

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

    def _substep(msg: str) -> None:
        if not log_callback:
            return
        try:
            if (os.environ.get("ENABLE_SUBSTEP_LOGS") or "").lower() in {"1", "true", "yes"}:
                log_callback(f"System1: {msg}")
        except Exception:
            pass

    _substep("enter prepare_data")
    # Fast path: reuse precomputed indicators
    if reuse_indicators and raw_data_dict:
        _substep("fast-path check start")
        try:
            # Early check - verify required indicators exist
            valid_data_dict, error_symbols = check_precomputed_indicators(
                raw_data_dict, SYSTEM1_REQUIRED_INDICATORS, "System1", skip_callback
            )

            if valid_data_dict:
                # Apply System1-specific filters
                prepared_dict = {}
                for symbol, df in valid_data_dict.items():
                    x = df.copy()

                    # Filter: Close>=5, DollarVolume20>25M
                    x["filter"] = (x["Close"] >= 5.0) & (x["dollarvolume20"] > 25_000_000)

                    # Setup: Filter + Close>SMA200 + ROC200>0
                    x["setup"] = x["filter"] & (x["Close"] > x["sma200"]) & (x["roc200"] > 0)

                    prepared_dict[symbol] = x

                _substep(f"fast-path processed symbols={len(prepared_dict)}")

                return prepared_dict

        except RuntimeError:
            # Re-raise error immediately if required indicators are missing
            raise
        except Exception:
            # Fall back to normal processing for other errors
            _substep("fast-path failed fallback to normal path")

    # Normal processing path: batch processing from symbol list
    if symbols:
        target_symbols = symbols
    elif raw_data_dict:
        target_symbols = list(raw_data_dict.keys())
    else:
        _substep("no symbols provided -> empty dict")
        return {}

    _substep(f"normal path start symbols={len(target_symbols)}")

    # Execute batch processing
    _substep("batch processing start")
    results, error_symbols = process_symbols_batch(
        target_symbols,
        _compute_indicators,
        batch_size=batch_size,
        use_process_pool=use_process_pool,
        max_workers=max_workers,
        progress_callback=progress_callback,
        log_callback=log_callback,
        skip_callback=skip_callback,
        system_name="System1",
    )
    _substep(f"batch processing done ok={len(results)} err={len(error_symbols)}")

    return results


def generate_candidates_system1(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int | None = None,
    progress_callback=None,
    log_callback=None,
    batch_size: int | None = None,
    latest_only: bool = False,
    **kwargs,
) -> tuple[dict, pd.DataFrame | None]:
    """System1 candidate generation (ROC200 descending ranking).

    Args:
        prepared_dict: Prepared data dictionary
        top_n: Number of top entries to extract
        progress_callback: Progress reporting callback
        log_callback: Log output callback

    Returns:
        (Daily candidate dictionary, Integrated candidate DataFrame)
    """
    if not prepared_dict:
        if log_callback:
            log_callback("System1: No data provided for candidate generation")
        return {}, None

    if top_n is None:
        top_n = 20  # Default value

    # Fast path: 当日（最新日）だけでランキングする用途 (today run) 向け最適化
    # full-history が必要なバックテストでは latest_only=False を指定して従来処理を維持
    if latest_only:
        try:
            rows: list[dict] = []
            date_counter: dict[pd.Timestamp, int] = {}
            debug_reasons: list[str] = []
            debug_enabled = (
                bool(os.environ.get("SYSTEM_DEBUG_VERBOSE")) or True
            )  # 常に有効(後で削除)
            for sym, df in prepared_dict.items():
                if df is None or df.empty:
                    if debug_enabled:
                        debug_reasons.append(f"{sym}:empty")
                    continue
                last_row = df.iloc[-1]
                # 優先: setup 列が True なら無条件に候補対象
                setup_flag = bool(last_row.get("setup", True))  # 無い場合 True (緩和)
                # フォールバック: setup False でも SMA25>SMA50 & filter 条件を満たせば許容
                if not setup_flag:
                    try:
                        sma25_v = last_row.get("sma25")
                        sma50_v = last_row.get("sma50")
                        if (
                            pd.isna(sma25_v)
                            or pd.isna(sma50_v)
                            or not (float(sma25_v) > float(sma50_v))
                        ):
                            if debug_enabled:
                                debug_reasons.append(f"{sym}:sma25<=sma50")
                            continue
                        # filter 列があれば True を要求（無ければ通す）
                        if ("filter" in last_row.index) and (not last_row.get("filter", False)):
                            if debug_enabled:
                                debug_reasons.append(f"{sym}:filterFalse")
                            continue
                    except Exception:
                        if debug_enabled:
                            debug_reasons.append(f"{sym}:fallbackException")
                        continue
                roc200_val = last_row.get("roc200", 0)
                try:
                    if pd.isna(roc200_val) or float(roc200_val) <= 0:
                        if debug_enabled:
                            try:
                                debug_reasons.append(f"{sym}:roc200<=0({roc200_val})")
                            except Exception:
                                pass
                        continue
                except Exception:
                    if debug_enabled:
                        debug_reasons.append(f"{sym}:roc200Err")
                    continue
                date_val = df.index[-1]
                date_counter[date_val] = date_counter.get(date_val, 0) + 1
                rows.append(
                    {
                        "symbol": sym,
                        "date": date_val,
                        "roc200": roc200_val,
                        "close": last_row.get("Close", 0),
                        "setup": bool(last_row.get("setup", False)),
                    }
                )
            if not rows:
                if log_callback:
                    log_callback(
                        "System1: latest_only fast-path produced 0 rows (after gating v2) — will fallback"
                    )
                    if debug_reasons and log_callback:
                        # 上位数件のみ
                        log_callback("System1: exclude_reasons=" + ", ".join(debug_reasons[:12]))
                # 明示的に通常パスへフォールバック
            else:
                df_all = pd.DataFrame(rows)
                try:
                    mode_date = max(date_counter.items(), key=lambda kv: kv[1])[0]
                    df_all = df_all[df_all["date"] == mode_date]
                except Exception:
                    pass
                df_all = df_all.sort_values("roc200", ascending=False, kind="stable").head(top_n)
                # 期待形式: date -> {symbol: {...各指標...}}
                by_date: dict[pd.Timestamp, dict[str, dict]] = {}
                for dt, sub in df_all.groupby("date"):
                    symbol_map: dict[str, dict] = {}
                    recs = sub.to_dict("records")
                    for rec in recs:
                        sym = rec.get("symbol")
                        if not sym:
                            continue
                        # symbol と date を除き残りを属性辞書に
                        symbol_map[str(sym)] = {
                            k: v for k, v in rec.items() if k not in {"symbol", "date"}
                        }
                    by_date[dt] = symbol_map
                if log_callback:
                    log_callback(
                        f"System1: latest_only fast-path -> {sum(len(v) for v in by_date.values())} candidates (symbols={len(rows)})"
                    )
                    if debug_enabled:
                        try:
                            first_dt = next(iter(by_date))
                            first_sym, first_payload = next(iter(by_date[first_dt].items()))
                            log_callback(
                                f"System1: first_candidate dt={first_dt} sym={first_sym} payload={first_payload}"
                            )
                        except Exception:
                            pass
                out_df = df_all.copy()
                return by_date, out_df
        except Exception as fast_err:
            if log_callback:
                log_callback(f"System1: fast-path failed -> fallback ({fast_err})")
            # 続けて従来パスへ
            pass

    # Aggregate all dates
    all_dates = set()
    for df in prepared_dict.values():
        if df is not None and not df.empty:
            all_dates.update(df.index)

    if not all_dates:
        if log_callback:
            log_callback("System1: No valid dates found in data")
        return {}, None

    all_dates = sorted(all_dates)

    candidates_by_date = {}
    all_candidates = []

    if log_callback:
        log_callback(f"System1: Generating candidates for {len(all_dates)} dates")

    # Execute ROC200 ranking by date
    for i, date in enumerate(all_dates):
        date_candidates = []

        for symbol, df in prepared_dict.items():
            try:
                if df is None or date not in df.index:
                    continue

                row = df.loc[date]

                # Check setup conditions
                if not row.get("setup", False):
                    continue

                # Get ROC200 value
                roc200_val = row.get("roc200", 0)
                if pd.isna(roc200_val) or roc200_val <= 0:
                    continue

                date_candidates.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "roc200": roc200_val,
                        "close": row.get("Close", 0),
                    }
                )

            except Exception:
                continue

        # Sort by ROC200 descending and extract top_n
        if date_candidates:
            date_candidates.sort(key=lambda x: x["roc200"], reverse=True)
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
        candidates_df = candidates_df.sort_values(["date", "roc200"], ascending=[True, False])
    else:
        candidates_df = None

    if log_callback:
        total_candidates = len(all_candidates)
        unique_dates = len(candidates_by_date)
        log_callback(
            f"System1: Generated {total_candidates} candidates across {unique_dates} dates"
        )

    return candidates_by_date, candidates_df


def get_total_days_system1(data_dict: dict[str, pd.DataFrame]) -> int:
    """Get total days count for System1 data.

    Args:
        data_dict: Data dictionary

    Returns:
        Maximum day count
    """
    return get_total_days(data_dict)


def generate_roc200_ranking_system1(
    data_dict: dict[str, pd.DataFrame], date: str, top_n: int = 20, log_callback=None
) -> list[dict]:
    """Generate ROC200-based ranking for a specific date.

    Args:
        data_dict: Dictionary of prepared data
        date: Target date (YYYY-MM-DD format)
        top_n: Number of top candidates to return
        log_callback: Optional logging callback

    Returns:
        List of candidate dictionaries with symbol, ROC200, and other metrics
    """
    if not data_dict:
        if log_callback:
            log_callback("System1: No data available for ranking")
        return []

    target_date = pd.to_datetime(date)
    candidates = []

    for symbol, df in data_dict.items():
        try:
            if df is None or target_date not in df.index:
                continue

            row = df.loc[target_date]

            # Check setup conditions
            if not row.get("setup", False):
                continue

            # Get ROC200 value
            roc200_val = row.get("roc200", 0)
            if pd.isna(roc200_val) or roc200_val <= 0:
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "roc200": float(roc200_val),
                    "close": float(row.get("Close", 0)),
                    "sma200": float(row.get("sma200", 0)),
                    "setup": bool(row.get("setup", False)),
                }
            )

        except Exception:
            continue

    # Sort by ROC200 descending and take top_n
    candidates.sort(key=lambda x: x["roc200"], reverse=True)
    result = candidates[:top_n]

    if log_callback:
        log_callback(f"System1: Generated {len(result)} ROC200 candidates for {date}")

    return result


__all__ = [
    "prepare_data_vectorized_system1",
    "generate_candidates_system1",
    "get_total_days_system1",
    "generate_roc200_ranking_system1",
]
