"""System5 core logic (Long mean-reversion with high ADX)."""

import os
import time

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, SMAIndicator
from ta.volatility import AverageTrueRange

from common.i18n import tr
from common.utils import get_cached_data, resolve_batch_size, BatchSizeMonitor

# Trading thresholds - Default values for business rules
DEFAULT_ATR_PCT_THRESHOLD = 0.04  # 4% ATR percentage threshold for filtering


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    df = get_cached_data(symbol)
    if df is None or df.empty:
        return symbol, None
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
    # 型を数値へ強制（ta ライブラリの安定化）
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 不正行の除外とインデックス整列
    x = df.copy()
    x = x.dropna(subset=[c for c in ("High", "Low", "Close") if c in x.columns])
    if not x.index.is_monotonic_increasing:
        try:
            x = x.sort_index()
        except Exception:
            pass
    if len(x) < 100:
        return symbol, None
    try:
        x["SMA100"] = SMAIndicator(x["Close"], window=100).sma_indicator()
        x["ATR10"] = AverageTrueRange(
            x["High"], x["Low"], x["Close"], window=10
        ).average_true_range()
        x["ADX7"] = ADXIndicator(x["High"], x["Low"], x["Close"], window=7).adx()
        x["RSI3"] = RSIIndicator(x["Close"], window=3).rsi()
        # Volume 欠損は 0 扱い
        vol = x["Volume"] if "Volume" in x.columns else pd.Series(0, index=x.index)
        x["AvgVolume50"] = vol.rolling(50).mean()
        x["DollarVolume50"] = (x["Close"] * vol).rolling(50).mean()
        # 0 除算ガード
        x["ATR_Pct"] = x["ATR10"].div(x["Close"].replace(0, pd.NA))
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
    # スキップ理由の内訳カウンタ
    skipped_insufficient_rows = 0
    skipped_missing_cols = 0
    skipped_calc_errors = 0
    missing_cols_examples: dict[str, int] = {}
    buffer: list[str] = []
    start_time = time.time()
    batch_monitor = BatchSizeMonitor(batch_size)
    batch_start = time.time()

    def _calc_indicators(src: pd.DataFrame) -> pd.DataFrame:
        x = src.copy()
        # 型を数値へ強制
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in x.columns:
                x[col] = pd.to_numeric(x[col], errors="coerce")
        # 不正行の除外と整列
        x = x.dropna(subset=[c for c in ("High", "Low", "Close") if c in x.columns])
        if not x.index.is_monotonic_increasing:
            try:
                x = x.sort_index()
            except Exception:
                pass
        if len(x) < 100:
            raise ValueError("insufficient rows")
        x["SMA100"] = SMAIndicator(x["Close"], window=100).sma_indicator()
        x["ATR10"] = AverageTrueRange(
            x["High"], x["Low"], x["Close"], window=10
        ).average_true_range()
        x["ADX7"] = ADXIndicator(x["High"], x["Low"], x["Close"], window=7).adx()
        x["RSI3"] = RSIIndicator(x["Close"], window=3).rsi()
        vol = x["Volume"] if "Volume" in x.columns else pd.Series(0, index=x.index)
        x["AvgVolume50"] = vol.rolling(50).mean()
        x["DollarVolume50"] = (x["Close"] * vol).rolling(50).mean()
        x["ATR_Pct"] = x["ATR10"].div(x["Close"].replace(0, pd.NA))
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
            # 事前チェック: 行数と必須列
            if len(df) < 100:
                skipped += 1
                skipped_insufficient_rows += 1
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
                continue
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            miss = [c for c in required_cols if c not in df.columns]
            if miss:
                skipped += 1
                skipped_missing_cols += 1
                for m in miss:
                    missing_cols_examples[m] = missing_cols_examples.get(m, 0) + 1
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
                continue
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
        except ValueError as e:
            skipped += 1
            skipped_calc_errors += 1
            # insufficient rows の ValueError を分類
            try:
                msg = str(e).lower()
                reason = "insufficient_rows" if "insufficient" in msg else "calc_error"
            except Exception:
                reason = "calc_error"
            if reason == "insufficient_rows":
                skipped_insufficient_rows += 1
                skipped_calc_errors -= 1  # 調整
            if skip_callback:
                try:
                    skip_callback(sym, reason)
                except Exception:
                    try:
                        skip_callback(f"{sym}: {reason}")
                    except Exception:
                        pass
        except Exception:
            skipped += 1
            skipped_calc_errors += 1
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
            # 追加の内訳（多い順に表示、ノイズ防止のため上位のみ）
            if skipped_insufficient_rows:
                try:
                    log_callback(f"  ├─ 行数不足(<100): {skipped_insufficient_rows} 件")
                except Exception:
                    pass
            if skipped_missing_cols:
                try:
                    # 欠落列の上位3件
                    top_missing = sorted(
                        missing_cols_examples.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                    details = ", ".join([f"{k}:{v}" for k, v in top_missing]) if top_missing else ""
                    log_callback(
                        f"  ├─ 必須列欠落: {skipped_missing_cols} 件"
                        + (f" ({details})" if details else "")
                    )
                except Exception:
                    pass
            if skipped_calc_errors:
                try:
                    log_callback(f"  └─ 計算エラー: {skipped_calc_errors} 件")
                except Exception:
                    pass
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
                # 翌営業日に補正
                try:
                    idx = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce").normalize())
                    pos = idx.searchsorted(ts, side="right")
                    if pos >= len(idx):
                        continue
                    entry_date = pd.to_datetime(idx[pos]).tz_localize(None)
                except Exception:
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
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dt.normalize()
        else:
            dates = pd.to_datetime(df.index).normalize()
        all_dates.update(dates)
    return len(all_dates)


__all__ = [
    "prepare_data_vectorized_system5",
    "generate_candidates_system5",
    "get_total_days_system5",
]
