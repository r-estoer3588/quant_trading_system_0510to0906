"""System4 core logic (Long trend low-vol pullback)."""

import os
import time

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange

from common.i18n import tr
from common.utils import get_cached_data, resolve_batch_size, BatchSizeMonitor


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    df = get_cached_data(symbol)
    if df is None or df.empty:
        return symbol, None
    df = df.copy(deep=False)
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

    x = df.copy(deep=False)
    if len(x) < 200:
        return symbol, None
    try:
        x["SMA200"] = SMAIndicator(x["Close"], window=200).sma_indicator()
        x["ATR40"] = AverageTrueRange(
            x["High"], x["Low"], x["Close"], window=40
        ).average_true_range()
        # pandas Series ベースでの対数リターン（型安定化）
        pct = x["Close"].pct_change()
        log_ret = pct.apply(lambda r: np.log1p(r) if pd.notnull(r) else r)
        x["HV50"] = log_ret.rolling(50).std() * np.sqrt(252) * 100
        x["RSI4"] = RSIIndicator(x["Close"], window=4).rsi()
        x["DollarVolume50"] = (x["Close"] * x["Volume"]).rolling(50).mean()
    except Exception:
        return symbol, None
    return symbol, x


def prepare_data_vectorized_system4(
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
    cache_dir = "data_cache/indicators_system4_cache"
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
    start_time = time.time()
    batch_monitor = BatchSizeMonitor(batch_size)
    batch_start = time.time()
    processed, skipped = 0, 0
    buffer: list[str] = []

    def _calc_indicators(src: pd.DataFrame) -> pd.DataFrame:
        # 不要なブロック統合を避けるため浅いコピー
        x = src.copy(deep=False)
        if len(x) < 200:
            raise ValueError("insufficient rows")
        x["SMA200"] = SMAIndicator(x["Close"], window=200).sma_indicator()
        x["ATR40"] = AverageTrueRange(
            x["High"], x["Low"], x["Close"], window=40
        ).average_true_range()
        # pandas Series ベースでの対数リターン（型安定化）
        pct = x["Close"].pct_change()
        log_ret = pct.apply(lambda r: np.log1p(r) if pd.notnull(r) else r)
        x["HV50"] = log_ret.rolling(50).std() * np.sqrt(252) * 100
        x["RSI4"] = RSIIndicator(x["Close"], window=4).rsi()
        x["DollarVolume50"] = (x["Close"] * x["Volume"]).rolling(50).mean()
        return x

    for sym, df in raw_data_dict.items():
        # 列名の大小文字差を吸収
        df = df.copy(deep=False)
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

        # 必須列チェック
        needed = {"Open", "High", "Low", "Close", "Volume"}
        miss = [c for c in needed if c not in df.columns]
        if miss:
            skipped += 1
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
            buffer.append(sym)
        except ValueError as e:
            skipped += 1
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
            skipped += 1
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
            batch_duration = time.time() - batch_start
            batch_size = batch_monitor.update(batch_duration)
            batch_start = time.time()
            try:
                log_callback(msg)
                log_callback(
                    tr(
                        "⏱️ batch time: {sec:.2f}s | next batch size: {size}",
                        sec=batch_duration,
                        size=batch_size,
                    )
                )
            except Exception:
                pass
            buffer.clear()

    if skipped > 0 and log_callback:
        try:
            log_callback(f"⚠️ データ不足/計算失敗でスキップ: {skipped} 件")
        except Exception:
            pass
    return result_dict


def generate_candidates_system4(
    prepared_dict: dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
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
    start_time = time.time()
    processed, skipped = 0, 0
    buffer: list[str] = []

    # 浅いコピーで十分（新規列追加のみ）
    spy_df = market_df.copy(deep=False)
    spy_df["SMA200"] = SMAIndicator(spy_df["Close"], window=200).sma_indicator()
    spy_df["spy_filter"] = (spy_df["Close"] > spy_df["SMA200"]).astype(int)

    for sym, df in prepared_dict.items():
        try:
            # メモリ節約のため浅いコピー
            x = df.copy(deep=False)
            cond_dv = x["DollarVolume50"] > 100_000_000
            cond_hv = x["HV50"].between(10, 40)
            x["filter"] = cond_dv & cond_hv
            x["setup"] = x["filter"] & (x["Close"] > x["SMA200"])

            setup_days = x[x["setup"]]
            if setup_days.empty:
                continue
            for date, row in setup_days.iterrows():
                ts = pd.to_datetime(pd.Index([date]))[0]
                if ts not in spy_df.index:
                    continue
                if int(spy_df.at[ts, "spy_filter"]) == 0:
                    continue
                # 翌営業日に補正
                try:
                    idx = pd.DatetimeIndex(pd.to_datetime(x.index, errors="coerce").normalize())
                    pos = idx.searchsorted(ts, side="right")
                    if pos >= len(idx):
                        continue
                    entry_date = pd.to_datetime(idx[pos]).tz_localize(None)
                except Exception:
                    entry_date = ts + pd.Timedelta(days=1)
                    if entry_date not in x.index:
                        continue
                rec = {
                    "symbol": sym,
                    "entry_date": entry_date,
                    "RSI4": row["RSI4"],
                    "ATR40": row["ATR40"],
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

    # rank by RSI4 ascending
    for date in list(candidates_by_date.keys()):
        ranked = sorted(candidates_by_date[date], key=lambda r: r["RSI4"])
        candidates_by_date[date] = ranked[: int(top_n)]

    if skipped > 0 and log_callback:
        try:
            log_callback(f"⚠️ 候補抽出中にスキップ: {skipped} 件")
        except Exception:
            pass
    return candidates_by_date, None


def get_total_days_system4(data_dict: dict[str, pd.DataFrame]) -> int:
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
    "prepare_data_vectorized_system4",
    "generate_candidates_system4",
    "get_total_days_system4",
]
