"""System6 fixed version - Uses pre-computed indicators from indicators_common.py"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from common.i18n import tr
from common.utils import BatchSizeMonitor, get_cached_data, is_today_run, resolve_batch_size
from common.utils_spy import resolve_signal_entry_date

SYSTEM6_BASE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# 既存インジケーターと整合性を保つ
SYSTEM6_FEATURE_COLUMNS = [
    "atr10",  # indicators_common.pyで計算済み
    "dollarvolume50",  # indicators_common.pyで計算済み
    "return_6d",  # indicators_common.pyで計算済み (Return6D → return_6d)
    "uptwodays",  # indicators_common.pyで計算済み (UpTwoDays → uptwodays)
    "filter",
    "setup",
]

SYSTEM6_ALL_COLUMNS = SYSTEM6_BASE_COLUMNS + SYSTEM6_FEATURE_COLUMNS
SYSTEM6_NUMERIC_COLUMNS = ["atr10", "dollarvolume50", "return_6d"]


def _prepare_system6_data_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """既存インジケーターを活用してSystem6データを準備（再計算なし）"""
    missing = [col for col in SYSTEM6_BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    x = df.copy()
    x = x.sort_index()
    if len(x) < 50:
        raise ValueError("insufficient rows")

    # indicators_common.pyで既に計算されているインジケーターをチェック
    required_indicators = ["atr10", "dollarvolume50", "return_6d", "uptwodays"]
    missing_indicators = [col for col in required_indicators if col not in x.columns]

    if missing_indicators:
        raise ValueError(f"missing pre-computed indicators: {', '.join(missing_indicators)}")

    try:
        # フィルタとセットアップのみ計算（軽量）
        x["filter"] = (x["Low"] >= 5) & (x["dollarvolume50"] > 10_000_000)
        x["setup"] = x["filter"] & (x["return_6d"] > 0.20) & x["uptwodays"]
    except Exception as exc:
        raise ValueError("calc_error") from exc

    x = x.dropna(subset=SYSTEM6_NUMERIC_COLUMNS)
    if x.empty:
        raise ValueError("insufficient rows")

    x = x.loc[~x.index.duplicated()].sort_index()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    x.index.name = "Date"
    return x


def prepare_data_fixed_system6(
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
    """固定版System6データ準備（既存インジケーター活用）"""

    result_dict: dict[str, pd.DataFrame] = {}
    raw_data_dict = raw_data_dict or {}

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

    for symbol, df in raw_data_dict.items():
        if df is None or df.empty:
            skipped += 1
            if skip_callback:
                try:
                    skip_callback(symbol, "empty_data")
                except Exception:
                    pass
            continue

        try:
            # 既存インジケーターを活用（再計算なし）
            prepared = _prepare_system6_data_from_frame(df)
            result_dict[symbol] = prepared
        except ValueError as e:
            if "missing pre-computed indicators" in str(e):
                if log_callback:
                    log_callback(f"[警告] {symbol}: {e}")
            skipped += 1
            if skip_callback:
                try:
                    skip_callback(symbol, "missing_indicators")
                except Exception:
                    pass
            continue
        except Exception:
            skipped += 1
            if skip_callback:
                try:
                    skip_callback(symbol, "calc_error")
                except Exception:
                    pass
            continue

        processed += 1
        buffer.append(symbol)

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
                "📊 System6 fixed progress: {done}/{total} | elapsed: {em}m{es}s / "
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

    return result_dict


def generate_candidates_system6_fixed(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
    batch_size: int | None = None,
) -> tuple[dict, pd.DataFrame | None]:
    """固定版System6候補生成（ベクトル化対応）"""

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

    for sym, df in prepared_dict.items():
        if df is None or df.empty:
            skipped += 1
            continue

        missing_cols = [c for c in SYSTEM6_ALL_COLUMNS if c not in df.columns]
        if missing_cols:
            skipped += 1
            continue

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
                entry_date = resolve_signal_entry_date(ts)
                if pd.isna(entry_date):
                    continue

                rec = {
                    "symbol": sym,
                    "entry_date": entry_date,
                    "entry_price": last_price,
                    "return_6d": row["return_6d"],  # Return6D → return_6d
                    "atr10": row["atr10"],
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
                "📊 System6 candidates progress: {done}/{total} | elapsed: {em}m{es}s / "
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

    # 候補をソート・制限
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

        # return_6dでソート（System6はショート戦略なので降順）
        df = df.sort_values("return_6d", ascending=False)
        total_candidates = len(df)
        df.loc[:, "rank"] = list(range(1, total_candidates + 1))
        df.loc[:, "rank_total"] = total_candidates
        limited = df.head(limit_n)
        candidates_by_date[date] = limited.to_dict("records")

    return candidates_by_date, None


def get_total_days_system6_fixed(data_dict: dict[str, pd.DataFrame]) -> int:
    """固定版System6日数カウント"""
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
    "prepare_data_fixed_system6",
    "generate_candidates_system6_fixed",
    "get_total_days_system6_fixed",
]
