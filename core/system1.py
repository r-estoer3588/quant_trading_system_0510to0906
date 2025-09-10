"""System1 core logic.

Provides data preparation, ROC200 ranking, and total-days helpers.
"""

import time
import pandas as pd

from common.utils import resolve_batch_size


def prepare_data_vectorized_system1(
    raw_data_dict,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
    batch_size: int | None = None,
    reuse_indicators=True,
    **kwargs,
):
    """System1 indicator computation (UI-agnostic)."""
    import os

    cache_dir = "data_cache/indicators_system1_cache"
    os.makedirs(cache_dir, exist_ok=True)
    total_symbols = len(raw_data_dict)
    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        batch_size = resolve_batch_size(total_symbols, batch_size)
    processed = 0
    symbol_buffer = []
    start_time = time.time()
    result_dict = {}

    for sym, df in raw_data_dict.items():
        if "Date" in df.columns:
            df = df.copy()
            df.index = pd.to_datetime(df["Date"]).dt.normalize()
        else:
            df = df.copy()
            df.index = pd.to_datetime(df.index).normalize()

        cache_path = os.path.join(cache_dir, f"{sym}.feather")
        cached: pd.DataFrame | None = None
        if reuse_indicators and os.path.exists(cache_path):
            try:
                cached = pd.read_feather(cache_path)
                cached["Date"] = pd.to_datetime(cached["Date"]).dt.normalize()
                cached.set_index("Date", inplace=True)
            except Exception:
                cached = None

        def _calc_indicators(src: pd.DataFrame) -> pd.DataFrame:
            dst = src.copy()
            dst["SMA25"] = dst["Close"].rolling(25).mean()
            dst["SMA50"] = dst["Close"].rolling(50).mean()
            dst["ROC200"] = dst["Close"].pct_change(200) * 100
            tr = pd.concat(
                [
                    dst["High"] - dst["Low"],
                    (dst["High"] - dst["Close"].shift()).abs(),
                    (dst["Low"] - dst["Close"].shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)
            dst["ATR20"] = tr.rolling(20).mean()
            dst["DollarVolume20"] = (dst["Close"] * dst["Volume"]).rolling(20).mean()
            dst["filter"] = (dst["Low"] >= 5) & (dst["DollarVolume20"] > 50_000_000)
            dst["setup"] = dst["filter"] & (dst["SMA25"] > dst["SMA50"])
            return dst

        if cached is not None and not cached.empty:
            last_date = cached.index.max()
            new_rows = df[df.index > last_date]
            if new_rows.empty:
                result_df = cached
            else:
                context_start = last_date - pd.Timedelta(days=200)
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
        processed += 1
        symbol_buffer.append(sym)

        if progress_callback:
            try:
                progress_callback(processed, total_symbols)
            except Exception:
                pass

        if (processed % batch_size == 0 or processed == total_symbols) and log_callback:
            elapsed = time.time() - start_time
            remaining = (elapsed / processed) * (total_symbols - processed)
            elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
            remain_min, remain_sec = divmod(int(remaining), 60)
            joined_syms = ", ".join(symbol_buffer)
            try:
                log_callback(
                    f"📊 指標計算: {processed}/{total_symbols} 件 完了"
                    f" | 経過: {elapsed_min}分{elapsed_sec}秒 / "
                    f"残り: 約 {remain_min}分{remain_sec}秒\n"
                    f"銘柄: {joined_syms}"
                )
            except Exception:
                pass
            symbol_buffer.clear()

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
    all_signals = []
    for symbol, df in data_dict.items():
        if "setup" not in df.columns or df["setup"].sum() == 0:
            continue
        sig_df = df[df["setup"]][["ROC200", "ATR20", "Open"]].copy()
        sig_df["symbol"] = symbol
        sig_df["entry_date"] = sig_df.index + pd.Timedelta(days=1)
        all_signals.append(sig_df.reset_index())

    if not all_signals:
        return {}, pd.DataFrame()

    all_signals_df = pd.concat(all_signals, ignore_index=True)

    if "SMA100" not in spy_df.columns:
        try:
            spy_df = spy_df.copy()
            spy_df["SMA100"] = spy_df["Close"].rolling(100).mean()
        except Exception:
            pass
    spy_df = spy_df[["Close", "SMA100"]].reset_index().rename(columns={"Date": "date"})

    merged = pd.merge_asof(
        all_signals_df.sort_values("Date"),
        spy_df.rename(
            columns={"Close": "Close_SPY", "SMA100": "SMA100_SPY"}
        ).sort_values("date"),
        left_on="Date",
        right_on="date",
    )
    merged = merged[merged["Close_SPY"] > merged["SMA100_SPY"]].copy()

    merged["entry_date_norm"] = merged["entry_date"].dt.normalize()
    grouped = merged.groupby("entry_date_norm")
    total_days = len(grouped)
    start_time = time.time()
    on_progress = kwargs.get("on_progress")
    on_log = kwargs.get("on_log")

    candidates_by_date = {}
    top_n = int(kwargs.get("top_n", 10))
    for i, (date, group) in enumerate(grouped, 1):
        top_df = group.nlargest(top_n, "ROC200")
        candidates_by_date[date] = top_df.to_dict("records")

        if on_progress:
            on_progress(i, total_days, start_time)
        if on_log and (i % 10 == 0 or i == total_days):
            elapsed = time.time() - start_time
            remain = elapsed / i * (total_days - i)
            on_log(
                f"📊 ROC200ランキング: {i}/{total_days} 日処理完了"
                f" | 経過: {int(elapsed // 60)}分{int(elapsed % 60)}秒"
                f" / 残り: 約 {int(remain // 60)}分{int(remain % 60)}秒"
            )

    return candidates_by_date, merged


__all__ = [
    "prepare_data_vectorized_system1",
    "get_total_days_system1",
    "generate_roc200_ranking_system1",
]
