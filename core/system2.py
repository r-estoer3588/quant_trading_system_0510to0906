"""System2 core logic (Short RSI spike) を共通化。

- インジケーター: RSI3, ADX7, ATR10, DollarVolume20, ATR_Ratio, TwoDayUp
- セットアップ条件: Close>5, DollarVolume20>25M, ATR_Ratio>0.03, RSI3>90, TwoDayUp
- 候補生成: ADX7 降順で top_n を日別抽出
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

from common.utils import get_cached_data, resolve_batch_size, BatchSizeMonitor


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Worker-side indicator calculation for a single symbol."""
    df = get_cached_data(symbol)
    if df is None or df.empty:
        return symbol, None

    # lower-case の OHLCV を許容
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

    base_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if base_cols:
        x = df[base_cols].copy()
    else:
        needed = [c for c in ["Close", "Open", "High", "Low"] if c in df.columns]
        x = df[needed].copy() if needed else df.copy(deep=False)

    if len(x) < 20:
        return symbol, None

    try:
        x["RSI3"] = RSIIndicator(x["Close"], window=3).rsi()
        x["ADX7"] = ADXIndicator(x["High"], x["Low"], x["Close"], window=7).adx()
        x["ATR10"] = AverageTrueRange(
            x["High"], x["Low"], x["Close"], window=10
        ).average_true_range()
    except Exception:
        return symbol, None

    if "Volume" in x.columns:
        x["DollarVolume20"] = (x["Close"] * x["Volume"]).rolling(window=20).mean()
    else:
        x["DollarVolume20"] = pd.Series(index=x.index, dtype=float)
    x["ATR_Ratio"] = x["ATR10"] / x["Close"]
    x["TwoDayUp"] = (x["Close"] > x["Close"].shift(1)) & (x["Close"].shift(1) > x["Close"].shift(2))
    x["filter"] = (x["Low"] >= 5) & (x["DollarVolume20"] > 25_000_000) & (x["ATR_Ratio"] > 0.03)
    x["setup"] = x["filter"] & (x["RSI3"] > 90) & x["TwoDayUp"]
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
        result_dict: dict[str, pd.DataFrame] = {}
        buffer: list[str] = []
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_indicators, sym): sym for sym in symbols}
            for i, fut in enumerate(as_completed(futures), 1):
                sym, df = fut.result()
                if df is not None:
                    result_dict[sym] = df
                    buffer.append(sym)
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
                    if buffer:
                        msg += f"銘柄: {', '.join(buffer)}"
                    try:
                        log_callback(msg)
                    except Exception:
                        pass
                    buffer.clear()
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
    buffer = []
    result_dict: dict[str, pd.DataFrame] = {}
    skipped_count = 0

    def _calc_indicators(src: pd.DataFrame) -> pd.DataFrame:
        base_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in src.columns]
        if base_cols:
            x = src[base_cols].copy()
        else:
            needed = [c for c in ["Close", "Open", "High", "Low"] if c in src.columns]
            x = src[needed].copy() if needed else src.copy(deep=False)
        if len(x) < 20:
            raise ValueError("insufficient rows")
        x["RSI3"] = RSIIndicator(x["Close"], window=3).rsi()
        x["ADX7"] = ADXIndicator(x["High"], x["Low"], x["Close"], window=7).adx()
        x["ATR10"] = AverageTrueRange(
            x["High"], x["Low"], x["Close"], window=10
        ).average_true_range()
        if "Volume" in x.columns:
            x["DollarVolume20"] = (x["Close"] * x["Volume"]).rolling(20).mean()
        else:
            x["DollarVolume20"] = pd.Series(index=x.index, dtype=float)
        x["ATR_Ratio"] = x["ATR10"] / x["Close"]
        x["TwoDayUp"] = (x["Close"] > x["Close"].shift(1)) & (
            x["Close"].shift(1) > x["Close"].shift(2)
        )
        x["filter"] = (x["Low"] >= 5) & (x["DollarVolume20"] > 25_000_000) & (x["ATR_Ratio"] > 0.03)
        x["setup"] = x["filter"] & (x["RSI3"] > 90) & x["TwoDayUp"]
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

        cache_path = os.path.join(cache_dir, f"{sym}.feather")
        cached: pd.DataFrame | None = None
        if reuse_indicators and os.path.exists(cache_path):
            try:
                cached = pd.read_feather(cache_path)
                cached["Date"] = pd.to_datetime(cached["Date"]).dt.normalize()
                cached.set_index("Date", inplace=True)
            except Exception:
                cached = None

        try:
            if cached is not None and not cached.empty:
                last_date = cached.index.max()
                new_rows = df[df.index > last_date]
                if new_rows.empty:
                    result_df = cached
                else:
                    context_start = last_date - pd.Timedelta(days=20)
                    recompute_src = df[df.index >= context_start]
                    recomputed = _calc_indicators(recompute_src)
                    recomputed = recomputed[recomputed.index > last_date]
                    result_df = pd.concat([cached, recomputed])
                    try:
                        result_df.reset_index().to_feather(cache_path)
                    except Exception:
                        pass
            else:
                result_df = _calc_indicators(df)
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
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce").normalize())
            base = pd.DatetimeIndex(pd.to_datetime(setup_df.index, errors="coerce").normalize())
            pos = idx.searchsorted(base, side="right")
            next_dates = pd.Series(pd.NaT, index=setup_df.index, dtype="datetime64[ns]")
            mask = (pos >= 0) & (pos < len(idx))
            if getattr(mask, "any", lambda: False)():
                next_vals = idx[pos[mask]]
                next_dates.loc[mask] = pd.to_datetime(next_vals).tz_localize(None)
            setup_df["entry_date"] = next_dates
        except Exception:
            setup_df["entry_date"] = pd.NaT

        mask_missing = setup_df["entry_date"].isna()
        if mask_missing.any():
            base_index = pd.to_datetime(setup_df.index, errors="coerce")
            fallback_values = []
            for base_dt in base_index[mask_missing.to_numpy()]:
                if pd.isna(base_dt):
                    fallback_values.append(pd.NaT)
                    continue
                try:
                    fallback = get_next_nyse_trading_day(pd.Timestamp(base_dt))
                except Exception:
                    fallback = pd.NaT
                fallback_values.append(fallback)
            setup_df.loc[mask_missing, "entry_date"] = fallback_values

        setup_df = setup_df.dropna(subset=["entry_date"])  # type: ignore[arg-type]
        all_signals.append(setup_df)

    if not all_signals:
        return {}, None

    all_df = pd.concat(all_signals)

    candidates_by_date = {}
    for date, group in all_df.groupby("entry_date"):
        ranked = group.sort_values("ADX7", ascending=False)
        candidates_by_date[date] = ranked.head(int(top_n)).to_dict("records")
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
