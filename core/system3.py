"""System3 core logic (Long mean-reversion)."""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange

from common.i18n import tr
from common.utils import get_cached_data, resolve_batch_size, BatchSizeMonitor

# Trading thresholds - Default values for business rules
DEFAULT_ATR_RATIO_THRESHOLD = 0.05  # 5% ATR ratio threshold for filtering


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

    x = df.copy()
    if len(x) < 150:
        return symbol, None
    try:
        x["SMA150"] = SMAIndicator(x["Close"], window=150).sma_indicator()
        x["ATR10"] = AverageTrueRange(
            x["High"], x["Low"], x["Close"], window=10
        ).average_true_range()
        x["Drop3D"] = -(x["Close"].pct_change(3))
        x["AvgVolume50"] = x["Volume"].rolling(50).mean()
        x["ATR_Ratio"] = x["ATR10"] / x["Close"]
        cond_price = x["Low"] >= 1
        cond_volume = x["AvgVolume50"] >= 1_000_000
        cond_atr = x["ATR_Ratio"] >= DEFAULT_ATR_RATIO_THRESHOLD
        x["filter"] = cond_price & cond_volume & cond_atr
        cond_close = x["Close"] > x["SMA150"]
        cond_drop = x["Drop3D"] >= 0.125
        x["setup"] = (x["filter"] & cond_close & cond_drop).astype(int)
    except Exception:
        return symbol, None
    return symbol, x


def prepare_data_vectorized_system3(
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
    cache_dir = "data_cache/indicators_system3_cache"
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
    buffer = []

    def _calc_indicators(src: pd.DataFrame) -> pd.DataFrame:
        x = src.copy()
        if len(x) < 150:
            raise ValueError("insufficient rows")
        x["SMA150"] = SMAIndicator(x["Close"], window=150).sma_indicator()
        x["ATR10"] = AverageTrueRange(
            x["High"], x["Low"], x["Close"], window=10
        ).average_true_range()
        x["Drop3D"] = -(x["Close"].pct_change(3))
        x["AvgVolume50"] = x["Volume"].rolling(50).mean()
        x["ATR_Ratio"] = x["ATR10"] / x["Close"]

        cond_price = x["Low"] >= 1
        cond_volume = x["AvgVolume50"] >= 1_000_000
        cond_atr = x["ATR_Ratio"] >= DEFAULT_ATR_RATIO_THRESHOLD
        x["filter"] = cond_price & cond_volume & cond_atr
        cond_close = x["Close"] > x["SMA150"]
        cond_drop = x["Drop3D"] >= 0.125
        cond_setup = x["filter"] & cond_close & cond_drop
        x["setup"] = cond_setup.astype(int)
        return x

    for sym, df in raw_data_dict.items():
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
                    context_start = last_date - pd.Timedelta(days=150)
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


def generate_candidates_system3(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
    progress_callback=None,
    log_callback=None,
    batch_size: int | None = None,
) -> tuple[dict, pd.DataFrame | None]:
    all_signals = []
    total = len(prepared_dict)
    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        batch_size = resolve_batch_size(total, batch_size)
    processed, skipped = 0, 0
    buffer = []
    start_time = time.time()

    for sym, df in prepared_dict.items():
        processed += 1
        if "setup" not in df.columns or not df["setup"].any():
            skipped += 1
            continue
        setup_df = df[df["setup"] == 1].copy()
        setup_df["symbol"] = sym
        # 翌営業日に補正
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
            setup_df = setup_df.dropna(subset=["entry_date"])  # type: ignore[arg-type]
        except Exception:
            setup_df["entry_date"] = pd.to_datetime(setup_df.index) + pd.Timedelta(days=1)
        setup_df = setup_df[["symbol", "entry_date", "Drop3D", "ATR10"]]
        all_signals.append(setup_df)
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

    if log_callback:
        try:
            log_callback(f"✅ 候補銘柄: {len(all_signals)} 件 / ⚠️ 候補対象外銘柄: {skipped} 件")
        except Exception:
            pass

    if not all_signals:
        return {}, None

    all_df = pd.concat(all_signals)
    candidates_by_date = {}
    for date, group in all_df.groupby("entry_date"):
        ranked = group.sort_values("Drop3D", ascending=False)
        candidates_by_date[date] = ranked.head(int(top_n)).to_dict("records")
    return candidates_by_date, None


def get_total_days_system3(data_dict: dict[str, pd.DataFrame]) -> int:
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
    "prepare_data_vectorized_system3",
    "generate_candidates_system3",
    "get_total_days_system3",
]
