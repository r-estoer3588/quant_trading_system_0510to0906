"""System6 core logic (Short mean-reversion momentum burst)."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

import pandas as pd
from ta.volatility import AverageTrueRange

from common.i18n import tr
from common.utils import BatchSizeMonitor, get_cached_data, resolve_batch_size, is_today_run
from common.utils_spy import resolve_signal_entry_date

SYSTEM6_BASE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
SYSTEM6_FEATURE_COLUMNS = [
    "ATR10",
    "DollarVolume50",
    "Return6D",
    "UpTwoDays",
    "filter",
    "setup",
]
SYSTEM6_ALL_COLUMNS = SYSTEM6_BASE_COLUMNS + SYSTEM6_FEATURE_COLUMNS
SYSTEM6_NUMERIC_COLUMNS = ["ATR10", "DollarVolume50", "Return6D"]


def _compute_indicators_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in SYSTEM6_BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    x = df.loc[:, SYSTEM6_BASE_COLUMNS].copy()
    x = x.sort_index()
    if len(x) < 50:
        raise ValueError("insufficient rows")
    try:
        x["ATR10"] = AverageTrueRange(
            x["High"], x["Low"], x["Close"], window=10
        ).average_true_range()
        x["DollarVolume50"] = (x["Close"] * x["Volume"]).rolling(50).mean()
        x["Return6D"] = x["Close"].pct_change(6)
        x["UpTwoDays"] = (x["Close"] > x["Close"].shift(1)) & (
            x["Close"].shift(1) > x["Close"].shift(2)
        )
        x["filter"] = (x["Low"] >= 5) & (x["DollarVolume50"] > 10_000_000)
        x["setup"] = x["filter"] & (x["Return6D"] > 0.20) & x["UpTwoDays"]
    except Exception as exc:
        raise ValueError("calc_error") from exc
    x = x.dropna(subset=SYSTEM6_NUMERIC_COLUMNS)
    if x.empty:
        raise ValueError("insufficient rows")
    x = x.loc[~x.index.duplicated()].sort_index()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    x.index.name = "Date"
    return x


def _load_system6_cache(cache_path: str) -> pd.DataFrame | None:
    if not os.path.exists(cache_path):
        return None
    try:
        cached = pd.read_feather(cache_path)
    except Exception:
        return None
    for col in ("Date", "date", "index"):
        if col in cached.columns:
            cached = cached.rename(columns={col: "Date"})
            break
    else:
        return None
    try:
        cached["Date"] = pd.to_datetime(cached["Date"]).dt.normalize()
        cached.set_index("Date", inplace=True)
    except Exception:
        return None
    cached = cached.sort_index()
    if not set(SYSTEM6_ALL_COLUMNS).issubset(cached.columns):
        return None
    cached = cached.loc[:, SYSTEM6_ALL_COLUMNS].copy()
    cached = cached.dropna(subset=SYSTEM6_NUMERIC_COLUMNS)
    if cached.empty:
        return None
    cached = cached.loc[~cached.index.duplicated()].sort_index()
    cached.index = pd.to_datetime(cached.index).tz_localize(None)
    cached.index.name = "Date"
    return cached


def _save_system6_cache(cache_path: str, df: pd.DataFrame) -> None:
    try:
        out = df.loc[:, SYSTEM6_ALL_COLUMNS].copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out.index.name = "Date"
        out.reset_index().to_feather(cache_path)
    except Exception:
        pass


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
    try:
        prepared = _compute_indicators_from_frame(df)
    except ValueError:
        return symbol, None
    except Exception:
        return symbol, None
    return symbol, prepared


def prepare_data_vectorized_system6(
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
    **kwargs,
) -> dict[str, pd.DataFrame]:
    cache_dir = "data_cache/indicators_system6_cache"
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
                        try:
                            today_mode = is_today_run()
                        except Exception:
                            today_mode = False
                        if not today_mode:
                            # Avoid logging very long symbol lists; show concise sample and count
                            sample = ", ".join(buffer[:10])
                            more = len(buffer) - len(buffer[:10])
                            if more > 0:
                                sample = f"{sample}, ...(+{more} more)"
                            msg += "\n" + tr("symbols: {names}", names=sample)
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
    skipped_insufficient_rows = 0
    skipped_missing_cols = 0
    skipped_calc_errors = 0
    buffer: list[str] = []

    def _calc_indicators(src: pd.DataFrame) -> pd.DataFrame:
        return _compute_indicators_from_frame(src)

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
            cached = _load_system6_cache(cache_path)

        try:
            if cached is not None and not cached.empty:
                last_date = cached.index.max()
                new_rows = df[df.index > last_date]
                if new_rows.empty:
                    result_df = cached
                else:
                    context_start = last_date - pd.Timedelta(days=50)
                    recompute_src = df[df.index >= context_start]
                    recomputed = _calc_indicators(recompute_src)
                    recomputed = recomputed[recomputed.index > last_date]
                    result_df = pd.concat([cached, recomputed])
                    result_df = result_df.loc[~result_df.index.duplicated(keep="last")].sort_index()
            else:
                result_df = _calc_indicators(df)
            _save_system6_cache(cache_path, result_df)
            result_dict[sym] = result_df
            buffer.append(sym)
        except ValueError as e:
            skipped += 1
            # 分類: insufficient_rows or calc_error
            try:
                msg = str(e).lower()
                reason = "insufficient_rows" if "insufficient" in msg else "calc_error"
            except Exception:
                reason = "calc_error"
            if reason == "insufficient_rows":
                skipped_insufficient_rows += 1
            else:
                skipped_calc_errors += 1
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
                try:
                    today_mode = is_today_run()
                except Exception:
                    today_mode = False
                if not today_mode:
                    # Show concise sample instead of full list
                    sample = ", ".join(buffer[:10])
                    more = len(buffer) - len(buffer[:10])
                    if more > 0:
                        sample = f"{sample}, ...(+{more} more)"
                    msg += "\n" + tr("symbols: {names}", names=sample)
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

    # 集計サマリーはログにのみ出力（skip_callback で集計を汚染しない）
    if skipped > 0 and log_callback:
        try:
            log_callback(f"⚠️ データ不足/計算失敗でスキップ: {skipped} 件")
            if skipped_insufficient_rows:
                try:
                    log_callback(f"  ├─ 行数不足(<50): {skipped_insufficient_rows} 件")
                except Exception:
                    pass
            if skipped_missing_cols:
                try:
                    log_callback(f"  ├─ 必須列欠落: {skipped_missing_cols} 件")
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


def generate_candidates_system6(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
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
    skipped_missing_cols = 0
    buffer: list[str] = []

    for sym, df in prepared_dict.items():
        # featherキャッシュの健全性チェック
        if df is None or df.empty:
            if log_callback:
                log_callback(f"[警告] {sym} のデータが空です（featherキャッシュ欠損）")
            skipped += 1
            continue
        missing_cols = [c for c in SYSTEM6_ALL_COLUMNS if c not in df.columns]
        if missing_cols:
            if log_callback:
                log_callback(
                    f"[警告] {sym} のデータに必須列が不足しています: {', '.join(missing_cols)}"
                )
            skipped += 1
            skipped_missing_cols += 1
            continue
        if df[SYSTEM6_NUMERIC_COLUMNS].isnull().any().any():
            if log_callback:
                log_callback(
                    f"[警告] {sym} のデータにNaNが含まれています（featherキャッシュ不完全）"
                )

        # last_price（直近終値）を取得
        last_price = None
        if "Close" in df.columns and not df["Close"].empty:
            last_price = df["Close"].iloc[-1]
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
                entry_date = resolve_signal_entry_date(ts)
                if pd.isna(entry_date):
                    continue
                rec = {
                    "symbol": sym,
                    "entry_date": entry_date,
                    "entry_price": last_price,
                    "Return6D": row["Return6D"],
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
                sample = ", ".join(buffer[:10])
                more = len(buffer) - len(buffer[:10])
                if more > 0:
                    sample = f"{sample}, ...(+{more} more)"
                msg += "\n" + tr("symbols: {names}", names=sample)
            try:
                log_callback(msg)
            except Exception:
                pass
            buffer.clear()

    limit_n = int(top_n)
    for date in list(candidates_by_date.keys()):
        rows = candidates_by_date.get(date, [])
        if not rows:
            candidates_by_date[date] = []
            continue
        df = pd.DataFrame(rows)
        if df.empty:
            candidates_by_date[date] = []
            continue
        df = df.sort_values("Return6D", ascending=False)
        total = len(df)
        df.loc[:, "rank"] = list(range(1, total + 1))
        df.loc[:, "rank_total"] = total
        limited = df.head(limit_n)
        candidates_by_date[date] = limited.to_dict("records")

    # 候補抽出の集計サマリーはログにのみ出力
    if skipped > 0 and log_callback:
        summary_lines = [f"⚠️ 候補抽出中にスキップ: {skipped} 件"]
        if skipped_missing_cols:
            summary_lines.append(f"  └─ 必須列欠落: {skipped_missing_cols} 件")
        try:
            for line in summary_lines:
                log_callback(line)
        except Exception:
            pass
    return candidates_by_date, None


def get_total_days_system6(data_dict: dict[str, pd.DataFrame]) -> int:
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
    "prepare_data_vectorized_system6",
    "generate_candidates_system6",
    "get_total_days_system6",
]
