"""System3 core logic (Long mean-reversion).

3-day drop mean-reversion strategy:
- Indicators: atr10, dollarvolume20, atr_ratio, drop3d (precomputed only)
- Setup conditions: Close>5, DollarVolume20>25M, atr_ratio>=0.05, drop3d>=0.125
- Candidate generation: drop3d descending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only

"""

from __future__ import annotations

from collections.abc import Callable
import os as _os
from typing import Any, cast

import pandas as pd

from common.batch_processing import process_symbols_batch
from common.system_common import check_precomputed_indicators, get_total_days
from common.system_constants import SYSTEM3_REQUIRED_INDICATORS
from common.system_setup_predicates import validate_predicate_equivalence
from common.utils import get_cached_data


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Check precomputed indicators and apply System3-specific filters.

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
        missing_indicators = [col for col in SYSTEM3_REQUIRED_INDICATORS if col not in df.columns]
        if missing_indicators:
            return symbol, None

        # Apply System3-specific filters and setup
        x = df.copy()

        # Filter: Close>=5, DollarVolume20>25M, ATR_Ratio>=0.05 (test override allowed)
        _atr_thr = 0.05
        try:
            _ov = _os.environ.get("MIN_ATR_RATIO_FOR_TEST")
            if _ov is not None:
                _atr_thr = float(str(_ov))
        except Exception:
            _atr_thr = 0.05
        x["filter"] = (
            (x["Close"] >= 5.0) & (x["dollarvolume20"] > 25_000_000) & (x["atr_ratio"] >= _atr_thr)
        )

        # Setup: Filter + drop3d>=0.125 (12.5% 3-day drop)
        x["setup"] = x["filter"] & (x["drop3d"] >= 0.125)

        return symbol, x

    except Exception:
        return symbol, None


def prepare_data_vectorized_system3(
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
    """System3 data preparation processing (3-day drop mean-reversion strategy).

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
                raw_data_dict, SYSTEM3_REQUIRED_INDICATORS, "System3", skip_callback
            )

            if valid_data_dict:
                # Apply System3-specific filters
                prepared_dict = {}
                for symbol, df in valid_data_dict.items():
                    x = df.copy()

                    # Filter: Close>=5, DollarVolume20>25M,
                    # ATR_Ratio>=0.05 (test override allowed)
                    _atr_thr = 0.05
                    try:
                        _ov = _os.environ.get("MIN_ATR_RATIO_FOR_TEST")
                        if _ov is not None:
                            _atr_thr = float(str(_ov))
                    except Exception:
                        _atr_thr = 0.05
                    x["filter"] = (
                        (x["Close"] >= 5.0)
                        & (x["dollarvolume20"] > 25_000_000)
                        & (x["atr_ratio"] >= _atr_thr)
                    )

                    # Setup: Filter + drop3d>=0.125 (12.5% 3-day drop)
                    x["setup"] = x["filter"] & (x["drop3d"] >= 0.125)

                    prepared_dict[symbol] = x

                if log_callback:
                    log_callback(f"System3: Fast-path processed {len(prepared_dict)} symbols")

                return prepared_dict

        except RuntimeError:
            # Re-raise error immediately if required indicators are missing
            raise
        except Exception:
            # Fall back to normal processing for other errors
            if log_callback:
                log_callback("System3: Fast-path failed, falling back to normal processing")

    # Normal processing path: batch processing from symbol list
    if symbols:
        target_symbols = symbols
    elif raw_data_dict:
        target_symbols = list(raw_data_dict.keys())
    else:
        if log_callback:
            log_callback("System3: No symbols provided, returning empty dict")
        return {}

    if log_callback:
        log_callback(f"System3: Starting normal processing for {len(target_symbols)} symbols")

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
        system_name="System3",
    )
    # Optional predicate equivalence validation (env gated)
    try:
        validate_predicate_equivalence(results, "3", log_fn=log_callback)
    except Exception:
        pass
    from typing import cast as _cast

    return _cast(dict[str, pd.DataFrame], results) if isinstance(results, dict) else {}


def generate_candidates_system3(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    batch_size: int | None = None,
    latest_only: bool = False,
    include_diagnostics: bool = False,
    **kwargs: Any,
) -> (
    tuple[dict[pd.Timestamp, dict[str, dict]], pd.DataFrame | None]
    | tuple[dict[pd.Timestamp, dict[str, dict]], pd.DataFrame | None, dict[str, Any]]
):
    """System3 candidate generation (drop3d descending ranking).

    Args:
        prepared_dict: Prepared data dictionary
        top_n: Number of top entries to extract
        progress_callback: Progress reporting callback
        log_callback: Log output callback

    Returns:
        (Daily candidate dictionary, Integrated candidate DataFrame)
    """
    # Initialize diagnostics dict
    diagnostics = {
        "ranking_source": None,
        "setup_predicate_count": 0,
        "predicate_only_pass_count": 0,
        "ranked_top_n_count": 0,
        "exclude_reasons": {},
        "mismatch_flag": 0,
    }

    if not prepared_dict:
        if log_callback:
            log_callback("System3: No data provided for candidate generation")
        return ({}, None, diagnostics) if include_diagnostics else ({}, None)

    if top_n is None:
        top_n = 20  # Default value

    if latest_only:
        # 最新日のみ対象。setup==True の銘柄を drop3d 降順で上位抽出
        rows: list[dict] = []
        date_counter: dict[pd.Timestamp, int] = {}
        # 許容営業日ラグを超えて target_date に一致しない銘柄を救済用に一時保持
        lagged_rows: list[dict] = []

        target_date = None
        try:
            maybe = kwargs.get("latest_mode_date")
            if maybe is not None:
                td = pd.to_datetime(str(maybe), errors="coerce")
                if (td is not None) and not pd.isna(td):
                    target_date = pd.Timestamp(td).normalize()
        except Exception:
            target_date = None

        def _to_series(obj: Any) -> pd.Series | None:
            try:
                if obj is None:
                    return None
                if isinstance(obj, pd.DataFrame):
                    return obj.iloc[-1]
                if isinstance(obj, pd.Series):
                    return obj
                return None
            except Exception:
                return None

        def _evaluate_row(
            row: pd.Series | None,
        ) -> tuple[bool, bool, bool, float, float, bool]:
            """Evaluate System3 setup conditions using predicate (no column dependency)."""
            if row is None:
                return False, False, False, float("nan"), float("nan"), False

            # Use predicate for setup evaluation
            try:
                from common.system_setup_predicates import system3_setup_predicate as _s3_pred
            except Exception:
                _s3_pred = None

            setup_flag = False
            if _s3_pred is not None:
                try:
                    setup_flag = bool(_s3_pred(row))
                except Exception:
                    setup_flag = False

            try:
                drop_val = float(row.get("drop3d", float("nan")))
            except Exception:
                drop_val = float("nan")

            try:
                atr_val = float(row.get("atr_ratio", float("nan")))
            except Exception:
                atr_val = float("nan")

            # Phase 2 filter already passed, no need to check filter column
            filter_flag = True
            final_flag = setup_flag

            try:
                override_drop = _os.environ.get("MIN_DROP3D_FOR_TEST")
                if (
                    not final_flag
                    and override_drop is not None
                    and filter_flag
                    and not pd.isna(drop_val)
                ):
                    thr = float(str(override_drop))
                    if drop_val >= thr:
                        final_flag = True
            except Exception:
                pass

            # predicate_flag is now same as setup_flag
            predicate_flag = setup_flag
            return (
                setup_flag,
                predicate_flag,
                final_flag,
                drop_val,
                atr_val,
                filter_flag,
            )

        # trading-day lag helper
        try:
            from common.utils_spy import calculate_trading_days_lag as _td_lag  # noqa: WPS433
        except Exception:
            _td_lag = None

        # tolerance days (orchestrator may pass max_date_lag_days)
        max_date_lag_days = 1
        try:
            lag_override = kwargs.get("max_date_lag_days")
            if lag_override is not None:
                max_date_lag_days = max(0, int(float(str(lag_override))))
        except Exception:
            max_date_lag_days = 1

        for sym, df in prepared_dict.items():
            try:
                if df is None or df.empty:
                    continue
                if target_date is not None:
                    if target_date in df.index:
                        last_row = _to_series(df.loc[target_date])
                        dt = target_date
                    else:
                        # allow fallback to latest if trading-day lag within tolerance
                        latest_idx_raw = df.index[-1]
                        latest_idx_norm = pd.Timestamp(str(latest_idx_raw)).normalize()
                        lag_days: int | None = None
                        try:
                            if _td_lag is not None:
                                lag_days = int(_td_lag(latest_idx_norm, target_date))
                            else:
                                lag_days = int((target_date - latest_idx_norm).days)
                        except Exception:
                            lag_days = None
                        if lag_days is not None and lag_days >= 0 and lag_days <= max_date_lag_days:
                            last_row = _to_series(df.loc[latest_idx_raw])
                            dt = target_date
                        else:
                            # 許容超過: rows には入れず、後段の不足補完用に保存
                            last_row = _to_series(df.loc[latest_idx_raw])
                            dt = target_date
                            if last_row is None:
                                continue
                            (
                                setup_col_ex,
                                _predicate_ex,
                                final_ok_ex,
                                drop_val_ex,
                                atr_val_ex,
                                _filter_ex,
                            ) = _evaluate_row(last_row)
                            if setup_col_ex:
                                diagnostics["setup_predicate_count"] += 1
                            if final_ok_ex and not setup_col_ex:
                                diagnostics["predicate_only_pass_count"] += 1
                                diagnostics["mismatch_flag"] = 1
                            if (not final_ok_ex) or pd.isna(drop_val_ex):
                                continue
                            try:
                                from common.utils_spy import (
                                    resolve_signal_entry_date as _resolve_entry_ex,
                                )

                                entry_dt_ex = _resolve_entry_ex(dt)
                            except Exception:
                                entry_dt_ex = None
                            atr_payload_ex = 0 if pd.isna(atr_val_ex) else atr_val_ex
                            lagged_rows.append(
                                {
                                    "symbol": sym,
                                    "date": dt,
                                    "entry_date": entry_dt_ex,
                                    "drop3d": drop_val_ex,
                                    "atr_ratio": atr_payload_ex,
                                    "close": last_row.get("Close", 0),
                                }
                            )
                            continue
                else:
                    last_row = _to_series(df.iloc[-1])
                    dt = pd.Timestamp(str(df.index[-1])).normalize()

                if last_row is None:
                    continue

                (
                    setup_col,
                    _predicate_flag,
                    final_ok,
                    drop_val,
                    atr_val,
                    _filter_flag,
                ) = _evaluate_row(last_row)

                if setup_col:
                    diagnostics["setup_predicate_count"] += 1
                if final_ok and not setup_col:
                    diagnostics["predicate_only_pass_count"] += 1
                    diagnostics["mismatch_flag"] = 1

                if not final_ok:
                    continue
                if pd.isna(drop_val):
                    continue

                date_counter[dt] = date_counter.get(dt, 0) + 1
                # 明示エントリー日（翌営業日）
                try:
                    from common.utils_spy import resolve_signal_entry_date as _resolve_entry

                    entry_dt = _resolve_entry(dt)
                except Exception:
                    entry_dt = None

                atr_payload = 0 if pd.isna(atr_val) else atr_val

                rows.append(
                    {
                        "symbol": sym,
                        "date": dt,
                        "entry_date": entry_dt,
                        "drop3d": drop_val,
                        "atr_ratio": atr_payload,
                        "close": last_row.get("Close", 0),
                    }
                )
            except Exception:
                continue

        if not rows:
            if log_callback:
                try:
                    # 代表サンプルを 1-2 件だけ出力して切り分けを容易にする
                    samples: list[str] = []
                    taken = 0
                    for s_sym, s_df in prepared_dict.items():
                        if s_df is None or getattr(s_df, "empty", True):
                            continue
                        try:
                            s_last = s_df.iloc[-1]
                            s_dt = pd.to_datetime(str(s_df.index[-1])).normalize()
                            (
                                s_setup,
                                _s_predicate,
                                s_final,
                                s_drop_val,
                                _s_atr,
                                _s_filter,
                            ) = _evaluate_row(s_last)
                            drop_txt = f"{s_drop_val:.4f}" if not pd.isna(s_drop_val) else "nan"
                            samples.append(
                                (
                                    f"{s_sym}: date={s_dt.date()} "
                                    f"setup_col={s_setup} final={s_final} "
                                    f"drop3d={drop_txt}"
                                )
                            )
                            taken += 1
                            if taken >= 2:
                                break
                        except Exception:
                            continue
                    if samples:
                        log_callback(
                            ("System3: DEBUG latest_only 0 candidates. " + " | ".join(samples))
                        )
                except Exception:
                    pass
                log_callback("System3: latest_only fast-path produced 0 rows")
            return ({}, None, diagnostics) if include_diagnostics else ({}, None)

        df_all = pd.DataFrame(rows)
        # top-off用に元の全候補を保持
        df_all_original = df_all.copy()
        if log_callback:
            log_callback(f"[DEBUG_S3_ROWS] rows={len(rows)} lagged_rows={len(lagged_rows)}")

        # target_date 優先でフィルタ。0件/不足時は安全フォールバックで補充する
        try:
            filtered = df_all
            final_label_date: pd.Timestamp | None = None
            if target_date is not None:
                filtered = df_all[df_all["date"] == target_date]
                final_label_date = target_date
                if filtered.empty and len(df_all) > 0:
                    # 最頻日の採用を試みる
                    try:
                        mode_date = max(date_counter.items(), key=lambda kv: kv[1])[0]
                    except Exception:
                        mode_date = None
                    if mode_date is not None:
                        filtered = df_all[df_all["date"] == mode_date]
                        final_label_date = mode_date
                    # それでも 0 件なら、全件を対象にして date を target_date へ上書き
                    if filtered.empty:
                        tmp = df_all.copy()
                        tmp.loc[:, "date"] = target_date
                        try:
                            from common.utils_spy import (
                                resolve_signal_entry_date as _resolve_entry_dt,
                            )

                            tmp.loc[:, "entry_date"] = _resolve_entry_dt(target_date)
                        except Exception:
                            tmp.loc[:, "entry_date"] = target_date
                        filtered = tmp
                        final_label_date = target_date
            else:
                mode_date = max(date_counter.items(), key=lambda kv: kv[1])[0]
                filtered = df_all[df_all["date"] == mode_date]
                final_label_date = mode_date
        except Exception:
            filtered = df_all
            final_label_date = None

        # ランキングして上位を確定
        ranked = filtered.sort_values("drop3d", ascending=False, kind="stable").copy()
        top_cut = ranked.head(top_n)

        # 足りない分を df_all_original + lagged_rows から補完（top-off）。date/entry_date を正規化。
        missing = max(0, int(top_n) - len(top_cut))
        if log_callback:
            log_callback(
                (
                    "[DEBUG_S3_TOPOFF] "
                    f"filtered={len(filtered)} top_cut={len(top_cut)} "
                    f"missing={missing} "
                    f"df_all_orig={len(df_all_original)} "
                    f"lagged={len(lagged_rows)}"
                )
            )
        # df_all_original または lagged_rows に補完候補がある場合に top-off を実行
        if missing > 0 and (len(df_all_original) > 0 or lagged_rows):
            try:
                exists = set(top_cut["symbol"].astype(str)) if not top_cut.empty else set()
                extras_pool = (
                    df_all_original.sort_values("drop3d", ascending=False, kind="stable")
                    .loc[~df_all_original["symbol"].astype(str).isin(exists)]
                    .copy()
                )
                # 許容ラグ超過の救済候補もプールに追加（重複symbolは除外）
                if lagged_rows:
                    lag_df = pd.DataFrame(lagged_rows)
                    if not lag_df.empty:
                        lag_df = lag_df.loc[~lag_df["symbol"].astype(str).isin(exists)]
                        extras_pool = pd.concat([extras_pool, lag_df], ignore_index=True)
                if not extras_pool.empty:
                    if final_label_date is None:
                        try:
                            final_label_date = (
                                target_date
                                if target_date is not None
                                else max(date_counter.items(), key=lambda kv: kv[1])[0]
                            )
                        except Exception:
                            final_label_date = None
                    if final_label_date is not None:
                        extras_pool.loc[:, "date"] = final_label_date
                        try:
                            from common.utils_spy import (
                                resolve_signal_entry_date as _resolve_entry_dt2,
                            )

                            extras_pool.loc[:, "entry_date"] = _resolve_entry_dt2(final_label_date)
                        except Exception:
                            extras_pool.loc[:, "entry_date"] = final_label_date
                    extras_take = extras_pool.head(missing)
                    top_cut = (
                        pd.concat([top_cut, extras_take], ignore_index=True)
                        .drop_duplicates(subset=["symbol"], keep="first")
                        .head(top_n)
                    )
            except Exception:
                pass

        df_all = top_cut
        diagnostics["ranked_top_n_count"] = len(df_all)
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
                symbol_map[sym_val] = payload
            by_date[dt] = symbol_map

        if log_callback:
            log_callback(
                (
                    "System3: latest_only fast-path -> "
                    f"{len(df_all)} candidates (symbols={len(rows)})"
                )
            )

        return (
            (by_date, df_all.copy(), diagnostics)
            if include_diagnostics
            else (by_date, df_all.copy())
        )

    # Aggregate all dates
    all_dates_set: set[pd.Timestamp] = set()
    for df in prepared_dict.values():
        if df is not None and not df.empty:
            all_dates_set.update(df.index)

    if not all_dates_set:
        if log_callback:
            log_callback("System3: No valid dates found in data")
        return ({}, None, diagnostics) if include_diagnostics else ({}, None)
    all_dates = sorted(all_dates_set)

    candidates_by_date: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    all_candidates: list[dict[str, Any]] = []

    if log_callback:
        log_callback(f"System3: Generating candidates for {len(all_dates)} dates")

    # Execute drop3d ranking by date
    for i, date in enumerate(all_dates):
        date_candidates: list[dict[str, Any]] = []
        for symbol, df in prepared_dict.items():
            try:
                if df is None or date not in df.index:
                    continue
                row = cast(pd.Series, df.loc[date])
                setup_val = bool(row.get("setup", False))
                from common.system_setup_predicates import system3_setup_predicate as _s3_pred

                pred_val = _s3_pred(row)
                if pred_val:
                    diagnostics["setup_predicate_count"] += 1
                if pred_val and not setup_val:
                    diagnostics["predicate_only_pass_count"] += 1
                    diagnostics["mismatch_flag"] = 1
                if not bool(setup_val):
                    continue
                drop3d_val = cast(Any, row.get("drop3d", 0))
                try:
                    if pd.isna(drop3d_val) or float(drop3d_val) < 0.125:
                        continue
                except Exception:
                    continue

                date_candidates.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "drop3d": drop3d_val,
                        "atr_ratio": row.get("atr_ratio", 0),
                        "close": row.get("Close", 0),
                    }
                )

            except Exception:
                continue

        # Sort by drop3d descending and extract top_n
        if date_candidates:
            date_candidates.sort(key=lambda x: x["drop3d"], reverse=True)
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
        candidates_df = candidates_df.sort_values(["date", "drop3d"], ascending=[True, False])
        diagnostics["ranking_source"] = "full_scan"
        try:
            last_dt = max(candidates_by_date.keys())
            diagnostics["ranked_top_n_count"] = len(candidates_by_date.get(last_dt, []))
        except Exception:
            diagnostics["ranked_top_n_count"] = 0
    else:
        candidates_df = None

    if log_callback:
        total_candidates = len(all_candidates)
        unique_dates = len(candidates_by_date)
        log_callback(
            ("System3: Generated " f"{total_candidates} candidates across {unique_dates} dates")
        )

    normalized: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    for dt, recs in candidates_by_date.items():
        out_symbol_map: dict[str, dict[str, Any]] = {}
        for rec_any in recs:
            rec_t: dict[str, Any] = rec_any
            sym_any = rec_t.get("symbol")
            if not isinstance(sym_any, str) or not sym_any:
                continue
            item_payload: dict[str, Any] = {
                str(k): v for k, v in rec_t.items() if k not in ("symbol", "date")
            }
            out_symbol_map[str(sym_any)] = item_payload
        normalized[dt] = out_symbol_map
    return (
        (normalized, candidates_df, diagnostics)
        if include_diagnostics
        else (normalized, candidates_df)
    )


def get_total_days_system3(data_dict: dict[str, pd.DataFrame]) -> int:
    """Get total days count for System3 data.

    Args:
        data_dict: Data dictionary

    Returns:
        Maximum day count
    """
    return int(get_total_days(data_dict))


__all__ = [
    "prepare_data_vectorized_system3",
    "generate_candidates_system3",
    "get_total_days_system3",
]
