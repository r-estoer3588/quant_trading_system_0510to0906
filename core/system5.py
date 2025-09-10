"""System5 core logic (Long mean-reversion with high ADX)."""

import os
import time

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, SMAIndicator
from ta.volatility import AverageTrueRange

from common.i18n import tr
from common.utils import get_cached_data, resolve_batch_size

# Trading thresholds - Default values for business rules
DEFAULT_ATR_PCT_THRESHOLD = 0.04  # 4% ATR percentage threshold for filtering


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    df = get_cached_data(symbol)
    if df is None or df.empty:
        return symbol, None
    x = df.copy()
    if len(x) < 100:
        return symbol, None
    try:
        x["SMA100"] = SMAIndicator(x["Close"], window=100).sma_indicator()
        x["ATR10"] = AverageTrueRange(
            x["High"], x["Low"], x["Close"], window=10
        ).average_true_range()
        x["ADX7"] = ADXIndicator(x["High"], x["Low"], x["Close"], window=7).adx()
        x["RSI3"] = RSIIndicator(x["Close"], window=3).rsi()
        x["AvgVolume50"] = x["Volume"].rolling(50).mean()
        x["DollarVolume50"] = (x["Close"] * x["Volume"]).rolling(50).mean()
        x["ATR_Pct"] = x["ATR10"] / x["Close"]
        x["filter"] = (
            (x["AvgVolume50"] > 500_000)
            & (x["DollarVolume50"] > 2_500_000)
            & (x["ATR_Pct"] > DEFAULT_ATR_PCT_THRESHOLD)
        )
        x["setup"] = (
            x["filter"]
            & (x["Close"] > x["SMA100"] + x["ATR10"])
            & (x["ADX7"] > 55)
            & (x["RSI3"] < 50)
        ).astype(int)
    except Exception:
        return symbol, None
    return symbol, x


def prepare_data_vectorized_system5(
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
    cache_dir = "data_cache/indicators_system5_cache"
    os.makedirs(cache_dir, exist_ok=True)
    result_dict: dict[str, pd.DataFrame] = {}
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
        buffer: list[str] = []
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_indicators, s): s for s in symbols}
            for i, fut in enumerate(as_completed(futures), 1):
                sym, df = fut.result()
                if df is not None:
                    result_dict[sym] = df
                    buffer.append(sym)
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
                    msg = tr(
                        "📊 indicators progress: {done}/{total} | elapsed: {em}m{es}s / "
                        "remain: ~{rm}m{rs}s",
                        done=i,
                        total=total,
                        em=em,
                        es=es,
                        rm=rm,
                        rs=rs,
                    )
                    if buffer:
                        msg += "\n" + tr("symbols: {names}", names=", ".join(buffer))
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
    processed, skipped = 0, 0
    buffer: list[str] = []
    start_time = time.time()

    def _calc_indicators(src: pd.DataFrame) -> pd.DataFrame:
        x = src.copy()
        if len(x) < 100:
            raise ValueError("insufficient rows")
        x["SMA100"] = SMAIndicator(x["Close"], window=100).sma_indicator()
        x["ATR10"] = AverageTrueRange(
            x["High"], x["Low"], x["Close"], window=10
        ).average_true_range()
        x["ADX7"] = ADXIndicator(x["High"], x["Low"], x["Close"], window=7).adx()
        x["RSI3"] = RSIIndicator(x["Close"], window=3).rsi()
        x["AvgVolume50"] = x["Volume"].rolling(50).mean()
        x["DollarVolume50"] = (x["Close"] * x["Volume"]).rolling(50).mean()
        x["ATR_Pct"] = x["ATR10"] / x["Close"]
        x["filter"] = (
            (x["AvgVolume50"] > 500_000)
            & (x["DollarVolume50"] > 2_500_000)
            & (x["ATR_Pct"] > DEFAULT_ATR_PCT_THRESHOLD)
        )
        x["setup"] = (
            x["filter"]
            & (x["Close"] > x["SMA100"] + x["ATR10"])
            & (x["ADX7"] > 55)
            & (x["RSI3"] < 50)
        ).astype(int)
        return x

    for sym, df in raw_data_dict.items():
        if "Date" in df.columns:
            df = df.copy()
            df.index = pd.Index(pd.to_datetime(df["Date"]).dt.normalize())
        else:
            df = df.copy()
            df.index = pd.Index(pd.to_datetime(df.index).normalize())

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
                    context_start = last_date - pd.Timedelta(days=100)
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
        except Exception:
            skipped += 1

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
            msg = tr(
                "📊 indicators progress: {done}/{total} | elapsed: {em}m{es}s / "
                "remain: ~{rm}m{rs}s",
                done=processed,
                total=total,
                em=em,
                es=es,
                rm=rm,
                rs=rs,
            )
            if buffer:
                msg += "\n" + tr("symbols: {names}", names=", ".join(buffer))
            try:
                log_callback(msg)
            except Exception:
                pass
            buffer.clear()

    if skipped > 0 and log_callback:
        try:
            log_callback(f"⚠️ データ不足/計算失敗でスキップ: {skipped} 件")
        except Exception:
            pass
    return result_dict


def generate_candidates_system5(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
    progress_callback=None,
    log_callback=None,
    batch_size: int | None = None,
) -> tuple[dict, pd.DataFrame | None]:
    candidates_by_date: dict[pd.Timestamp, list] = {}
    total = len(prepared_dict)
    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        batch_size = resolve_batch_size(total, batch_size)
    processed, skipped = 0, 0
    buffer: list[str] = []
    start_time = time.time()

    for sym, df in prepared_dict.items():
        try:
            if "setup" not in df.columns or not df["setup"].any():
                skipped += 1
                continue
            setup_days = df[df["setup"] == 1]
            if setup_days.empty:
                skipped += 1
                continue
            for date, row in setup_days.iterrows():
                ts = pd.to_datetime(pd.Index([date]))[0]
                entry_date = ts + pd.Timedelta(days=1)
                if entry_date not in df.index:
                    continue
                rec = {
                    "symbol": sym,
                    "entry_date": entry_date,
                    "ADX7": row["ADX7"],
                    "ATR10": row["ATR10"],
                }
                candidates_by_date.setdefault(entry_date, []).append(rec)
        except Exception:
            skipped += 1

        processed += 1
        buffer.append(sym)
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
            msg = tr(
                "📊 candidates progress: {done}/{total} | elapsed: {em}m{es}s / "
                "remain: ~{rm}m{rs}s",
                done=processed,
                total=total,
                em=em,
                es=es,
                rm=rm,
                rs=rs,
            )
            if buffer:
                msg += "\n" + tr("symbols: {names}", names=", ".join(buffer))
            try:
                log_callback(msg)
            except Exception:
                pass
            buffer.clear()

    for date in list(candidates_by_date.keys()):
        ranked = sorted(candidates_by_date[date], key=lambda r: r["ADX7"], reverse=True)
        candidates_by_date[date] = ranked[: int(top_n)]

    if skipped > 0 and log_callback:
        try:
            log_callback(f"⚠️ 候補抽出中にスキップ: {skipped} 件")
        except Exception:
            pass
    return candidates_by_date, None


def get_total_days_system5(data_dict: dict[str, pd.DataFrame]) -> int:
    all_dates = set()
    for df in data_dict.values():
        if df is None or df.empty:
            continue
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"]).dt.normalize()
        else:
            dates = pd.to_datetime(df.index).normalize()
        all_dates.update(dates)
    return len(all_dates)


__all__ = [
    "prepare_data_vectorized_system5",
    "generate_candidates_system5",
    "get_total_days_system5",
]
