"""System1 core logic.

Provides data preparation, ROC200 ranking, and total-days helpers.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import pandas as pd

from common.utils import (
    BatchSizeMonitor,
    describe_dtype,
    drop_duplicate_columns,
    get_cached_data,
    resolve_batch_size,
)
from common.utils_spy import resolve_signal_entry_date

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _rename_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
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
        x.rename(columns=rename_map, inplace=True)
    return x


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        idx = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    else:
        idx = pd.to_datetime(df.index, errors="coerce").normalize()
    if idx is None:
        raise ValueError("invalid_date_index")
    try:
        if pd.isna(idx).all():
            raise ValueError("invalid_date_index")
    except Exception:
        pass
    x = df.copy()
    x.index = pd.Index(idx)
    x.index.name = "Date"
    x = x[~x.index.isna()]
    try:
        x = x.sort_index()
    except Exception:
        pass
    try:
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
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=[c for c in ("High", "Low", "Close") if c in x.columns])
    return x


def _compute_indicators_frame(df: pd.DataFrame) -> pd.DataFrame:
    """No-op indicator computation - check precomputed indicators exist."""
    x = df.copy()
    
    # Required precomputed indicators (lowercase, from indicators_common)
    required_indicators = ["sma25", "sma50", "roc200", "atr20", "dollarvolume20"]
    
    # Check if all required indicators exist - if not, raise error
    missing_indicators = [col for col in required_indicators if col not in x.columns]
    if missing_indicators:
        raise ValueError(f"missing precomputed indicators: {missing_indicators}")
    
    # Use precomputed indicators only (no calculation)
    # Create strategy-specific derived columns
    x["filter"] = (x["Low"] >= 5) & (x["dollarvolume20"] > 50_000_000)
    x["setup"] = x["filter"] & (x["sma25"] > x["sma50"])
    return x


def _compute_indicators(
    symbol: str,
    cache_dir: str,
    reuse_indicators: bool,
) -> tuple[str, pd.DataFrame | None]:
    """Check precomputed indicators for a single symbol - no computation."""
    import os

    df = get_cached_data(symbol)
    if df is None or df.empty:
        return symbol, None

    # 子プロセスから親へ簡易進捗を送る（存在すれば）
    try:
        q = globals().get("_PROGRESS_QUEUE")
        if q is not None:
            try:
                q.put((symbol, 0))
            except Exception:
                pass
    except Exception:
        pass

    # 正規化: 日付インデックス・並び順・重複排除・型
    try:
        if "Date" in df.columns:
            idx = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        elif "date" in df.columns:
            idx = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        else:
            idx = pd.to_datetime(df.index, errors="coerce").normalize()
        df.index = pd.Index(idx)
        df = df[~df.index.isna()].sort_index()
        try:
            if getattr(df.index, "has_duplicates", False):
                df = df[~df.index.duplicated(keep="last")]
        except Exception:
            pass
    except Exception:
        return symbol, None

    # Required precomputed indicators (lowercase, from indicators_common)
    required_indicators = ["sma25", "sma50", "roc200", "atr20", "dollarvolume20"]
    
    # Check if all required indicators exist
    missing_indicators = [col for col in required_indicators if col not in df.columns]
    if missing_indicators:
        return symbol, None  # Early exit if any indicator missing

    try:
        # Use existing processing logic but with precomputed indicators only
        x = _prepare_source_frame(df)
        x = _compute_indicators_frame(x)
    except Exception:
        return symbol, None

    # 完了を親に伝える
    try:
        q = globals().get("_PROGRESS_QUEUE")
        if q is not None:
            try:
                q.put((symbol, 100))
            except Exception:
                pass
    except Exception:
        pass

    return symbol, x
                df = df[~df.index.duplicated(keep="last")]
        except Exception:
            pass
        # OHLCV を数値化
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass
    except Exception:
        # 失敗時も最低限のフォールバック
        try:
            df = df.sort_index()
        except Exception:
            pass

    date_series = pd.to_datetime(df.index, errors="coerce").normalize()
    latest_date = date_series.max()
    cache_path = os.path.join(cache_dir, f"{symbol}_{latest_date.date()}.feather")

    if reuse_indicators and os.path.exists(cache_path):
        try:
            cached = pd.read_feather(cache_path)
            if cached is not None and not cached.isnull().any().any():
                # 安全のため、日付インデックスを正規化し、昇順・重複除去
                if "Date" in cached.columns:
                    cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce").dt.normalize()
                    cached = (
                        cached.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
                    )
                    cached = cached.set_index("Date")
                else:
                    try:
                        idx = pd.to_datetime(cached.index, errors="coerce").normalize()
                        cached.index = pd.Index(idx)
                        cached = cached[~cached.index.isna()].sort_index()
                    except Exception:
                        pass
                try:
                    if getattr(cached.index, "has_duplicates", False):
                        cached = cached[~cached.index.duplicated(keep="last")]
                except Exception:
                    pass
                return symbol, cached
        except Exception:
            pass

    try:
        prepared = _prepare_source_frame(df)
    except ValueError:
        return symbol, None
    except Exception:
        return symbol, None

    prepared = _compute_indicators_frame(prepared)

    latest_df = prepared[date_series == latest_date]
    try:
        latest_df.reset_index(drop=True).to_feather(cache_path)
    except Exception:
        pass

    # 完了を親に伝える
    try:
        q = globals().get("_PROGRESS_QUEUE")
        if q is not None:
            try:
                q.put((symbol, 100))
            except Exception:
                pass
    except Exception:
        pass

    return symbol, prepared


def prepare_data_vectorized_system1(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
    batch_size: int | None = None,
    reuse_indicators: bool = True,
    *,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    **kwargs,
):
    """System1 indicator computation (UI-agnostic).

    ``use_process_pool`` が True の場合、各シンボルを ProcessPoolExecutor で並列処理し、
    キャッシュを各プロセスが直接読み込む。
    ``raw_data_dict`` が None の場合は ``symbols`` からキャッシュを取得する。
    """
    import os

    # Fast-path for today-mode: if frames already include the indicators from
    # rolling/shared precompute, just ensure filter/setup exist and return.
    try:
        if reuse_indicators and isinstance(raw_data_dict, dict) and raw_data_dict:
            required = {"sma25", "sma50", "roc200", "atr20", "dollarvolume20"}
            out_fast: dict[str, pd.DataFrame] = {}
            missing: dict[str, pd.DataFrame] = {}

            for sym, df in raw_data_dict.items():
                try:
                    if df is None or df.empty:
                        missing[sym] = df
                        continue
                    x = _rename_ohlcv(df)
                    # normalize index
                    try:
                        if "Date" in x.columns:
                            x.index = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                        else:
                            x.index = pd.to_datetime(x.index, errors="coerce").normalize()
                        x = x[~x.index.isna()].sort_index()
                    except Exception:
                        pass
                    have = set(x.columns)
                    if not {"Close", "High", "Low", "Volume"}.issubset(have):
                        missing[sym] = df
                        continue
                    if not required.issubset(have):
                        missing[sym] = df
                        continue
                    # derive filter/setup if absent
                    if "filter" not in x.columns:
                        try:
                            x["filter"] = (x["Low"] >= 5) & (x["dollarvolume20"] > 50_000_000)
                        except Exception:
                            x["filter"] = False
                    if "setup" not in x.columns:
                        try:
                            x["setup"] = x["filter"] & (x["sma25"] > x["sma50"])
                        except Exception:
                            x["setup"] = False
                    out_fast[str(sym)] = x
                except Exception:
                    missing[str(sym)] = df

            if len(out_fast) == len(raw_data_dict):
                return out_fast
            result_dict: dict[str, pd.DataFrame] = {}
            result_dict.update(out_fast)
            if missing:
                computed = prepare_data_vectorized_system1(
                    missing,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                    skip_callback=skip_callback,
                    batch_size=batch_size,
                    reuse_indicators=False,
                    symbols=list(missing.keys()),
                    use_process_pool=use_process_pool,
                    max_workers=max_workers,
                    **kwargs,
                )
                result_dict.update(computed)
            return result_dict
    except Exception:
        pass

    cache_dir = "data_cache/indicators_system1_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # normalize inputs for both branches
    raw_data_dict = raw_data_dict or {}
    default_total_symbols = len(raw_data_dict)
    # ensure total_symbols is defined for progress callbacks in either branch
    if symbols is not None and len(symbols) > 0:
        total_symbols = len(symbols)
    else:
        total_symbols = default_total_symbols

    if use_process_pool:
        if symbols is None:
            symbols = list(raw_data_dict.keys()) if raw_data_dict else []
        total_symbols = len(symbols)
        if batch_size is None:
            try:
                from config.settings import get_settings

                batch_size = get_settings(create_dirs=False).data.batch_size
            except Exception:
                batch_size = 100
            batch_size = resolve_batch_size(total_symbols, batch_size)

        result_dict: dict[str, pd.DataFrame] = {}
        symbol_buffer: list[str] = []
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_indicators, sym, cache_dir, reuse_indicators): sym
                for sym in symbols
            }
            for i, fut in enumerate(as_completed(futures), 1):
                sym = futures[fut]
                try:
                    sym_r, df = fut.result()
                    # 念のため返却シンボル優先
                    sym = sym_r or sym
                    if df is not None:
                        result_dict[sym] = df
                        symbol_buffer.append(sym)
                except Exception as e:
                    # 1件の失敗で全体を止めない
                    if "skip_callback" in kwargs:
                        try:
                            cb = kwargs.get("skip_callback")
                            if callable(cb):
                                try:
                                    cb(sym, f"calc_error:{e}")
                                except Exception:
                                    cb(f"{sym}: calc_error:{e}")
                        except Exception:
                            pass

                if progress_callback:
                    try:
                        progress_callback(i, total_symbols)
                    except Exception:
                        pass

                # ensure batch_size is int for modulus
                try:
                    _bs = int(batch_size) if batch_size is not None else 0
                except Exception:
                    _bs = 0
                if (_bs and (i % _bs == 0 or i == total_symbols)) and log_callback:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / i) * (total_symbols - i) if i else 0
                    em, es = divmod(int(elapsed), 60)
                    rm, rs = divmod(int(remaining), 60)
                    joined_syms = ", ".join(symbol_buffer)
                    try:
                        # split into multiple shorter calls to avoid long single-line strings
                        log_callback(f"📊 指標計算: {i}/{total_symbols} 件 完了")
                        log_callback(f"経過: {em}分{es}秒 / 残り: 約 {rm}分{rs}秒")
                        log_callback(f"銘柄: {joined_syms}")
                    except Exception:
                        pass
                    symbol_buffer.clear()
            if log_callback:
                try:
                    elapsed = time.time() - start_time
                    em, es = divmod(int(elapsed), 60)
                    log_callback(f"📊 指標計算: {len(result_dict)}/{total_symbols} 件 完了")
                    log_callback(f"経過: {em}分{es}秒")
                except Exception:
                    pass
    # batch monitor to adjust batch sizes over time in non-parallel mode
    try:
        _init_bs = int(batch_size) if batch_size is not None else 0
    except Exception:
        _init_bs = 0
    batch_monitor = BatchSizeMonitor(_init_bs)
    if not use_process_pool:
        total_symbols = default_total_symbols
    processed = 0
    symbol_buffer: list[str] = []
    start_time = time.time()
    batch_start = time.time()
    result_dict: dict[str, pd.DataFrame] = {}

    def _on_symbol_done(symbol: str | None = None, *, include_in_buffer: bool = False) -> None:
        nonlocal processed, batch_size, batch_start
        if include_in_buffer and symbol:
            symbol_buffer.append(symbol)
        processed += 1
        if progress_callback:
            try:
                progress_callback(processed, total_symbols)
            except Exception:
                pass
        try:
            _bs = int(batch_size) if batch_size is not None else 0
        except Exception:
            _bs = 0
        if (_bs and (processed % _bs == 0 or processed == total_symbols)) and log_callback:
            elapsed = time.time() - start_time
            remaining = (elapsed / processed) * (total_symbols - processed) if processed else 0
            em, es = divmod(int(elapsed), 60)
            rm, rs = divmod(int(remaining), 60)
            joined_syms = ", ".join(symbol_buffer)
            try:
                log_callback(f"📊 指標計算: {processed}/{total_symbols} 件 完了")
                log_callback(f"経過: {em}分{es}秒 / 残り: 約 {rm}分{rs}秒")
                log_callback(f"銘柄: {joined_syms}")
            except Exception:
                pass
            batch_duration = time.time() - batch_start
            try:
                batch_size = batch_monitor.update(batch_duration)
            except Exception:
                pass
            batch_start = time.time()
            symbol_buffer.clear()

    for sym, df in raw_data_dict.items():
        df = _rename_ohlcv(df)

        try:
            base_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
            if base_cols:
                base_nan_rate = df[base_cols].isnull().mean().mean()
            else:
                base_nan_rate = df.isnull().mean().mean() if df.size > 0 else 0.0
            if base_nan_rate >= 0.45:
                msg = f"⚠️ {sym} cache: OHLCV欠損率高 ({base_nan_rate:.2%})"
                if log_callback:
                    log_callback(msg)
                if skip_callback:
                    skip_callback(sym, msg)
                _on_symbol_done()
                continue
            if base_nan_rate > 0.20 and log_callback:
                log_callback(f"⚠️ {sym} cache: OHLCV欠損率注意 ({base_nan_rate:.2%})")

            indicator_cols = [
                c
                for c in df.columns
                if c not in base_cols
                and str(c).lower() not in {"date", "symbol"}
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            if indicator_cols:
                indicator_nan_rate = df[indicator_cols].isnull().mean().mean()
                if indicator_nan_rate > 0.60 and log_callback:
                    log_callback(f"⚠️ {sym} cache: 指標NaN率高 ({indicator_nan_rate:.2%})")

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    series_like = df[col]
                    if not pd.api.types.is_numeric_dtype(series_like):
                        dtype_repr = describe_dtype(series_like)
                        msg = f"⚠️ {sym} cache: {col}型不一致 ({dtype_repr})"
                        if log_callback:
                            log_callback(msg)
                        if skip_callback:
                            skip_callback(sym, msg)
            for col in ["Close", "High", "Low"]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    if (vals <= 0).all():
                        msg = f"⚠️ {sym} cache: {col}全て非正値"
                        if log_callback:
                            log_callback(msg)
                        if skip_callback:
                            skip_callback(sym, msg)
        except Exception as e:
            msg = f"⚠️ {sym} cache: 健全性チェック失敗 ({e})"
            if log_callback:
                log_callback(msg)
            if skip_callback:
                skip_callback(sym, msg)
            _on_symbol_done()
            continue

        cache_path = os.path.join(cache_dir, f"{sym}.feather")
        cached: pd.DataFrame | None = None
        if reuse_indicators and os.path.exists(cache_path):
            try:
                cached = pd.read_feather(cache_path)
                cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce").dt.normalize()
                cached = cached.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
                cached.set_index("Date", inplace=True)
                if getattr(cached.index, "has_duplicates", False):
                    cached = cached[~cached.index.duplicated(keep="last")]
            except Exception:
                cached = None

        try:
            prepared_df = _prepare_source_frame(df)
        except ValueError as exc:
            if skip_callback:
                reason_raw = str(exc)
                if reason_raw.startswith("missing_cols:"):
                    reason = reason_raw
                elif "invalid_date_index" in reason_raw:
                    reason = "invalid_date_index"
                else:
                    reason = "calc_error"
                try:
                    skip_callback(sym, reason)
                except Exception:
                    try:
                        skip_callback(f"{sym}: {reason}")
                    except Exception:
                        pass
            _on_symbol_done()
            continue
        except Exception as exc:
            if skip_callback:
                try:
                    skip_callback(sym, f"calc_error:{exc}")
                except Exception:
                    try:
                        skip_callback(f"{sym}: calc_error")
                    except Exception:
                        pass
            _on_symbol_done()
            continue

        try:
            if cached is not None and not cached.empty:
                last_date = cached.index.max()
                new_rows = prepared_df[prepared_df.index > last_date]
                if new_rows.empty:
                    result_df = cached
                else:
                    context_start = last_date - pd.Timedelta(days=200)
                    recompute_src = prepared_df[prepared_df.index >= context_start]
                    recomputed = _compute_indicators_frame(recompute_src)
                    recomputed = recomputed[recomputed.index > last_date]
                    result_df = pd.concat([cached, recomputed])
                    try:
                        result_df = result_df.sort_index()
                        if getattr(result_df.index, "has_duplicates", False):
                            result_df = result_df[~result_df.index.duplicated(keep="last")]
                        result_df.reset_index().to_feather(cache_path)
                    except Exception:
                        pass
            else:
                result_df = _compute_indicators_frame(prepared_df)
                try:
                    result_df.reset_index().to_feather(cache_path)
                except Exception:
                    pass
            result_dict[sym] = result_df
            _on_symbol_done(sym, include_in_buffer=True)
        except ValueError as exc:
            if skip_callback:
                try:
                    msg = str(exc).lower()
                    reason = "insufficient_rows" if "insufficient" in msg else "calc_error"
                    skip_callback(sym, reason)
                except Exception:
                    try:
                        skip_callback(f"{sym}: insufficient_rows")
                    except Exception:
                        pass
            _on_symbol_done()
        except Exception as exc:
            if skip_callback:
                try:
                    skip_callback(sym, f"calc_error:{exc}")
                except Exception:
                    try:
                        skip_callback(f"{sym}: calc_error")
                    except Exception:
                        pass
            _on_symbol_done()
    return result_dict


def get_total_days_system1(data_dict):
    """Return the total number of unique dates across prepared data."""
    all_dates = set()
    for df in data_dict.values():
        if df is None or df.empty:
            continue
        if "Date" in df.columns:
            date_series = pd.to_datetime(df["Date"]).dt.normalize()
        else:
            date_series = pd.to_datetime(df.index).normalize()
        all_dates.update(date_series)

    return len(sorted(all_dates))


def generate_roc200_ranking_system1(data_dict: dict, spy_df: pd.DataFrame, **kwargs):
    """Generate daily ROC200 ranking filtered by SPY trend."""
    on_progress = kwargs.get("on_progress")
    on_log = kwargs.get("on_log")
    all_signals = []
    for symbol, df in data_dict.items():
        if "setup" not in df.columns or df["setup"].sum() == 0:
            continue
        sig_df = df[df["setup"]][["ROC200", "ATR20", "Open"]].copy()
        sig_df["symbol"] = symbol
        idx_norm = pd.to_datetime(sig_df.index, errors="coerce").normalize()
        sig_df["Date"] = idx_norm
        # last_price（直近終値）を取得
        last_price = None
        if "Close" in df.columns and not df["Close"].empty:
            last_price = df["Close"].iloc[-1]
        sig_df["entry_price"] = last_price
        sig_df["entry_date"] = sig_df["Date"].map(resolve_signal_entry_date)
        sig_df = sig_df.dropna(subset=["entry_date"])
        all_signals.append(sig_df.reset_index(drop=True))

    if not all_signals:
        return {}, pd.DataFrame()

    all_signals_df = pd.concat(all_signals, ignore_index=True)

    # SPY 側の列を整備（大小文字・date 列を堅牢化）
    if "SMA100" not in spy_df.columns:
        try:
            spy_df = spy_df.copy()
            # Close 列が lower な場合の補正
            if "Close" not in spy_df.columns and "close" in spy_df.columns:
                spy_df["Close"] = spy_df["close"]
            spy_df["SMA100"] = spy_df["Close"].rolling(100).mean()
        except Exception:
            pass

    spy = spy_df.copy()
    # date 列の生成
    date_col = None
    if "Date" in spy.columns:
        spy["date"] = pd.to_datetime(spy["Date"], errors="coerce")
        date_col = "date"
    elif "date" in spy.columns:
        spy["date"] = pd.to_datetime(spy["date"], errors="coerce")
        date_col = "date"
    else:
        # index が日時の場吁E
        try:
            idx = pd.to_datetime(spy.index, errors="coerce")
            try:
                cond_any = pd.notna(idx).any()
            except Exception:
                try:
                    cond_any = idx.notna().any()
                except Exception:
                    cond_any = False
            if cond_any:
                spy = spy.reset_index().rename(columns={spy.index.name or "index": "date"})
                spy["date"] = pd.to_datetime(spy["date"], errors="coerce")
                date_col = "date"
        except Exception:
            pass
    # 必要列に絞ってソート
    if date_col is None:
        # date が無空Eならマージ不能なので空返却
        return {}, pd.DataFrame()
    spy = spy[["date", "Close", "SMA100"]].sort_values("date")
    log_duplicates = (lambda message: on_log(message)) if callable(on_log) else None
    spy = drop_duplicate_columns(
        spy,
        log_callback=log_duplicates,
        context="System1 ROC200: SPY系列",
    )

    merged = pd.merge_asof(
        all_signals_df.sort_values("Date"),
        spy.rename(columns={"Close": "Close_SPY", "SMA100": "SMA100_SPY"}),
        left_on="Date",
        right_on="date",
    )
    # NOTE: SPY のトレンド判定 (Close_SPY > SMA100_SPY) は
    # 戦略の「セットアップ」フェーズで評価するため、ここでは候補抽出
    # 側での絞り込みを行わない（UI/前処理側でセットアップ判定する）。

    merged["entry_date_norm"] = merged["entry_date"].dt.normalize()

    merged = drop_duplicate_columns(
        merged,
        log_callback=log_duplicates,
        context="System1 ROC200: merge結果",
    )

    grouped = merged.groupby("entry_date_norm")
    total_days = len(grouped)
    start_time = time.time()

    candidates_by_date = {}
    top_n = int(kwargs.get("top_n", 10))
    for i, (date, group) in enumerate(grouped, 1):
        top_df = group.nlargest(top_n, "ROC200")
        date_repr = date.date() if hasattr(date, "date") else date
        top_df = drop_duplicate_columns(
            top_df,
            log_callback=log_duplicates,
            context=f"System1 ROC200: {date_repr} Top{top_n}",
        )
        candidates_by_date[date] = top_df.to_dict("records")

        if on_progress:
            on_progress(i, total_days, start_time)
        if on_log and (i % 10 == 0 or i == total_days):
            elapsed = time.time() - start_time
            remain = elapsed / i * (total_days - i)
            on_log(
                f"📊 ROC200ランキング: {i}/{total_days} 日処理完了",
                f" | 経過: {int(elapsed // 60)}分{int(elapsed % 60)}秒",
                f" / 残り: 約 {int(remain // 60)}分{int(remain % 60)}秒",
            )

    return candidates_by_date, merged


__all__ = [
    "prepare_data_vectorized_system1",
    "get_total_days_system1",
    "generate_roc200_ranking_system1",
]
