"""System2 core logic (Short RSI spike) を共通化。

- インジケーター: RSI3, ADX7, ATR10, DollarVolume20, ATR_Ratio, TwoDayUp
- セットアップ条件: Close>5, DollarVolume20>25M, ATR_Ratio>0.03, RSI3>90, TwoDayUp
- 候補生成: ADX7 降順で top_n を日別抽出
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

from common.utils import BatchSizeMonitor, get_cached_data, resolve_batch_size
from common.utils_spy import resolve_signal_entry_date


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Worker-side: no computation, only check precomputed indicators exist."""
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

    # Normalize OHLCV column names
    rename_map = {}
    for low, up in (
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ):
        if low in df.columns and up not in df.columns:
            rename_map[low] = up
    if rename_map:
        df = df.rename(columns=rename_map)

    # Required precomputed indicators (lowercase, from indicators_common)
    required_indicators = ["rsi3", "adx7", "atr10", "dollarvolume20", "atr_ratio", "twodayup"]
    
    # Check if all required indicators exist
    missing_indicators = [col for col in required_indicators if col not in df.columns]
    if missing_indicators:
        return symbol, None  # Early exit if any indicator missing

    # Copy only necessary columns
    needed_cols = ["Open", "High", "Low", "Close", "Volume"] + required_indicators
    available_cols = [col for col in needed_cols if col in df.columns]
    x = df[available_cols].copy()

    if len(x) < 20:
        return symbol, None

    # No indicator calculation - use precomputed only
    # Create strategy-specific derived columns using lowercase names
    x["filter"] = (x["Low"] >= 5) & (x["dollarvolume20"] > 25_000_000) & (x["atr_ratio"] > 0.03)
    x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]

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


def prepare_data_vectorized_system2(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    *,
    progress_callback=None,
    log_callback=None,
    batch_size: int | None = None,
    reuse_indicators: bool = True,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    skip_callback=None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    # Fast-path for today-mode: if incoming frames already have the needed
    # indicator columns (from rolling/shared precompute), avoid recomputation
    # and just derive lightweight filter/setup columns.
    try:
        if reuse_indicators and isinstance(raw_data_dict, dict) and raw_data_dict:
            required = {"rsi3", "adx7", "atr10", "dollarvolume20", "atr_ratio", "twodayup"}
            out_fast: dict[str, pd.DataFrame] = {}
            missing: dict[str, pd.DataFrame] = {}

            for sym, df in raw_data_dict.items():
                try:
                    if df is None or df.empty:
                        missing[sym] = df
                        continue
                    x = df.copy()
                    # normalize OHLCV upper-case if lower exists
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
                    # index normalize for safety
                    try:
                        if "Date" in x.columns:
                            x.index = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                        else:
                            x.index = pd.to_datetime(x.index, errors="coerce").normalize()
                        x = x[~x.index.isna()]
                        x = x.sort_index()
                    except Exception:
                        pass
                    have = set(x.columns)
                    # Need core OHLC columns as well
                    if not {"Close", "High", "Low"}.issubset(have):
                        missing[sym] = df
                        continue
                    if not required.issubset(have):
                        missing[sym] = df
                        continue
                    # derive strategy-specific flags if absent
                    if "filter" not in x.columns:
                        try:
                            x["filter"] = (
                                (x["Low"] >= 5)
                                & (x["dollarvolume20"] > 25_000_000)
                                & (x["atr_ratio"] > 0.03)
                            )
                        except Exception:
                            x["filter"] = False
                    if "setup" not in x.columns:
                        try:
                            x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]
                        except Exception:
                            x["setup"] = False
                    out_fast[str(sym)] = x
                except Exception:
                    missing[str(sym)] = df

            # If we could satisfy all, return immediately without indicator logs
            if len(out_fast) == len(raw_data_dict):
                return out_fast
            # Otherwise, compute only for missing symbols and merge
            # 結果辞書（fast-path + 不足分計算結果）
            result_dict: dict[str, pd.DataFrame] = dict(out_fast)
            if missing:
                computed = prepare_data_vectorized_system2(
                    missing,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                    batch_size=batch_size,
                    reuse_indicators=False,
                    symbols=list(missing.keys()),
                    use_process_pool=use_process_pool,
                    max_workers=max_workers,
                    skip_callback=skip_callback,
                    **kwargs,
                )
                result_dict.update(computed)
            return result_dict
    except Exception:
        # fall back to normal path on any issue
        pass
    cache_dir = "data_cache/indicators_system2_cache"
    os.makedirs(cache_dir, exist_ok=True)
    raw_data_dict = raw_data_dict or {}
    if use_process_pool:
        if symbols is None:
            symbols = list(raw_data_dict.keys())
        total = len(symbols)
        if batch_size is None:
            try:
                from config.settings import get_settings

                batch_size = get_settings(create_dirs=False).data.batch_size
            except Exception:
                batch_size = 100
            batch_size = resolve_batch_size(total, batch_size)
        # プロセスプール用の結果辞書（再定義を避けるため新しい局所名を使用）
        result_dict = {}
        pool_buffer: list[str] = []
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_indicators, sym): sym for sym in symbols}
            for i, fut in enumerate(as_completed(futures), 1):
                sym, df = fut.result()
                if df is not None:
                    result_dict[sym] = df
                    pool_buffer.append(sym)
                else:
                    if skip_callback:
                        try:
                            skip_callback(sym, "pool_skipped")
                        except Exception:
                            try:
                                skip_callback(f"{sym}: pool_skipped")
                            except Exception:
                                pass
                if progress_callback:
                    try:
                        progress_callback(i, total)
                    except Exception:
                        pass
                if (i % batch_size == 0 or i == total) and log_callback:
                    elapsed = time.time() - start_time
                    remain = (elapsed / i) * (total - i) if i else 0
                    em, es = divmod(int(elapsed), 60)
                    rm, rs = divmod(int(remain), 60)
                    msg = (
                        f"📊 インジケーター計算 {i}/{total} 件 完了 | "
                        f"経過: {em}分{es}秒 / 残り: 約{rm}分{rs}秒\n"
                    )
                    if pool_buffer:
                        msg += f"銘柄: {', '.join(pool_buffer)}"
                    try:
                        log_callback(msg)
                    except Exception:
                        pass
                    pool_buffer.clear()
        return result_dict

    total = len(raw_data_dict)
    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        batch_size = resolve_batch_size(total, batch_size)
    processed = 0
    start_time = time.time()
    batch_monitor = BatchSizeMonitor(batch_size)
    batch_start = time.time()
    buffer: list[str] = []  # serial path buffer
    result_dict: dict[str, pd.DataFrame] = {}
    skipped_count = 0

    def _calc_indicators(src: pd.DataFrame) -> pd.DataFrame:
        base_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in src.columns]
        if base_cols:
            x = src[base_cols].copy()
        else:
            needed = [c for c in ["Close", "Open", "High", "Low"] if c in src.columns]
            x = src[needed].copy() if needed else src.copy(deep=False)
        # Check required columns exist in base dataframe
        if len(x) < 20:
            raise ValueError("insufficient rows")
        
        # Required precomputed indicators (lowercase, from indicators_common)
        required_indicators = ["rsi3", "adx7", "atr10", "dollarvolume20", "atr_ratio", "twodayup"]
        
        # Check if all required indicators exist - if not, skip this symbol
        missing_indicators = [col for col in required_indicators if col not in x.columns]
        if missing_indicators:
            raise ValueError(f"missing precomputed indicators: {missing_indicators}")
        
        # Use precomputed indicators only (no calculation)
        # Create strategy-specific derived columns
        x["filter"] = (x["Low"] >= 5) & (x["dollarvolume20"] > 25_000_000) & (x["atr_ratio"] > 0.03)
        x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]
        return x

    for sym, df in raw_data_dict.items():
        # 列名の大小文字差を吸収
        df = df.copy()
        rename_map = {}
        for low, up in (
            ("open", "Open"),
            ("high", "High"),
            ("low", "Low"),
            ("close", "Close"),
            ("volume", "Volume"),
        ):
            if low in df.columns and up not in df.columns:
                rename_map[low] = up
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        if "Date" in df.columns:
            df.index = pd.Index(pd.to_datetime(df["Date"]).dt.normalize())
        elif "date" in df.columns:
            df.index = pd.Index(pd.to_datetime(df["date"]).dt.normalize())
        else:
            df.index = pd.Index(pd.to_datetime(df.index).normalize())

        # 必須列の存在チェック（Volume はあれば使用）
        req = {"Open", "High", "Low", "Close"}
        miss = [c for c in req if c not in df.columns]
        if miss:
            skipped_count += 1
            if skip_callback:
                try:
                    skip_callback(sym, f"missing_cols:{','.join(miss)}")
                except Exception:
                    try:
                        skip_callback(f"{sym}: missing_cols:{','.join(miss)}")
                    except Exception:
                        pass
            processed += 1
            if progress_callback:
                try:
                    progress_callback(processed, total)
                except Exception:
                    pass
            continue

            # Check for required precomputed indicators (lowercase)
            required_indicators = ["rsi3", "adx7", "atr10", "dollarvolume20", "atr_ratio", "twodayup"]
            missing_indicators = [col for col in required_indicators if col not in df.columns]
            
            if missing_indicators:
                # Skip symbol if required precomputed indicators are missing
                skipped_count += 1
                if skip_callback:
                    try:
                        skip_callback(sym, f"missing_precomputed:{','.join(missing_indicators)}")
                    except Exception:
                        try:
                            skip_callback(f"{sym}: missing_precomputed:{','.join(missing_indicators)}")
                        except Exception:
                            pass
                processed += 1
                if progress_callback:
                    try:
                        progress_callback(processed, total)
                    except Exception:
                        pass
                continue

            # Use only precomputed indicators - no calculation
            needed_cols = base_cols + required_indicators
            available_cols = [col for col in needed_cols if col in df.columns]
            x = df[available_cols].copy()
            x = x.sort_index()
            
            if len(x) < 20:
                skipped_count += 1
                if skip_callback:
                    try:
                        skip_callback(sym, "insufficient_rows")
                    except Exception:
                        try:
                            skip_callback(f"{sym}: insufficient_rows")
                        except Exception:
                            pass
                processed += 1
                if progress_callback:
                    try:
                        progress_callback(processed, total)
                    except Exception:
                        pass
                continue

            # Create strategy-specific derived columns using precomputed indicators
            x["filter"] = (x["Low"] >= 5) & (x["dollarvolume20"] > 25_000_000) & (x["atr_ratio"] > 0.03)
            x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]

            result_df = x
            try:
                result_df.reset_index().to_feather(cache_path)
            except Exception:
                pass

            result_dict[sym] = result_df
            buffer.append(sym)
        except ValueError as e:
            skipped_count += 1
            if skip_callback:
                try:
                    msg = str(e).lower()
                    reason = "insufficient_rows" if "insufficient" in msg else "calc_error"
                    skip_callback(sym, reason)
                except Exception:
                    try:
                        skip_callback(f"{sym}: insufficient_rows")
                    except Exception:
                        pass
        except Exception:
            skipped_count += 1
            if skip_callback:
                try:
                    skip_callback(sym, "calc_error")
                except Exception:
                    try:
                        skip_callback(f"{sym}: calc_error")
                    except Exception:
                        pass

        processed += 1

        if progress_callback:
            try:
                progress_callback(processed, total)
            except Exception:
                pass
        if (processed % batch_size == 0 or processed == total) and log_callback:
            elapsed = time.time() - start_time
            remain = (elapsed / processed) * (total - processed) if processed else 0
            em, es = divmod(int(elapsed), 60)
            rm, rs = divmod(int(remain), 60)
            msg = (
                f"📊 インジケーター計算 {processed}/{total} 件 完了 | "
                f"経過: {em}分{es}秒 / 残り: 約{rm}分{rs}秒\n"
            )
            if buffer:
                msg += f"銘柄: {', '.join(buffer)}"
            batch_duration = time.time() - batch_start
            batch_size = batch_monitor.update(batch_duration)
            batch_start = time.time()
            try:
                log_callback(msg)
                log_callback(f"⏱️ バッチ時間: {batch_duration:.2f}秒 | 次バッチサイズ: {batch_size}")
            except Exception:
                pass
            buffer.clear()

    if skipped_count > 0 and log_callback:
        try:
            log_callback(f"⚠️ データ不足/計算失敗でスキップ: {skipped_count} 件")
        except Exception:
            pass

    return result_dict


def generate_candidates_system2(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
) -> tuple[dict, pd.DataFrame | None]:
    """セットアップ通過銘柄を日別に ADX7 降順で抽出。
    返却: (candidates_by_date, merged_df=None)
    """
    all_signals = []
    for sym, df in prepared_dict.items():
        if "setup" not in df.columns or not df["setup"].any():
            continue
        setup_df = df[df["setup"]].copy()
        setup_df["symbol"] = sym
        # last_price（直近終値）を取得
        last_price = None
        if "Close" in df.columns and not df["Close"].empty:
            last_price = df["Close"].iloc[-1]
        setup_df["entry_price"] = last_price
        base_dates = pd.to_datetime(setup_df.index, errors="coerce").to_series(index=setup_df.index)
        setup_df["entry_date"] = base_dates.map(resolve_signal_entry_date)
        setup_df = setup_df.dropna(subset=["entry_date"])
        all_signals.append(setup_df)

    if not all_signals:
        return {}, None

    all_df = pd.concat(all_signals)

    # entry_date 単位のシグナル候補: date -> list[dict]
    candidates_by_date: dict[pd.Timestamp, list[dict]] = {}
    for date, group in all_df.groupby("entry_date"):
        ranked = group.sort_values("ADX7", ascending=False).copy()
        total = len(ranked)
        if total == 0:
            candidates_by_date[date] = []
            continue
        ranked.loc[:, "rank"] = range(1, total + 1)
        ranked.loc[:, "rank_total"] = total
        top_ranked = ranked.head(int(top_n))
        candidates_by_date[date] = top_ranked.to_dict("records")
    return candidates_by_date, None


def get_total_days_system2(data_dict: dict[str, pd.DataFrame]) -> int:
    """データ中の日数ユニーク数。"""
    all_dates = set()
    for df in data_dict.values():
        if df is None or df.empty:
            continue
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"]).dt.normalize()
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dt.normalize()
        else:
            dates = pd.to_datetime(df.index).normalize()
        all_dates.update(dates)
    return len(all_dates)


__all__ = [
    "prepare_data_vectorized_system2",
    "generate_candidates_system2",
    "get_total_days_system2",
]
