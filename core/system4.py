"""System4 core logic (Long trend low-vol pullback)."""

from typing import Dict, Tuple
import time
import numpy as np
import pandas as pd
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from common.i18n import tr


def prepare_data_vectorized_system4(
    raw_data_dict: Dict[str, pd.DataFrame],
    *,
    progress_callback=None,
    log_callback=None,
    batch_size: int = 50,
) -> Dict[str, pd.DataFrame]:
    result_dict: Dict[str, pd.DataFrame] = {}
    total = len(raw_data_dict)
    start_time = time.time()
    processed, skipped = 0, 0
    buffer: list[str] = []

    for sym, df in raw_data_dict.items():
        x = df.copy()
        if len(x) < 200:
            skipped += 1
            processed += 1
            continue
        try:
            x["SMA200"] = SMAIndicator(x["Close"], window=200).sma_indicator()
            x["ATR40"] = AverageTrueRange(x["High"], x["Low"], x["Close"], window=40).average_true_range()
            x["HV50"] = (
                np.log(x["Close"] / x["Close"].shift(1)).rolling(50).std() * np.sqrt(252) * 100
            )
            x["RSI4"] = RSIIndicator(x["Close"], window=4).rsi()
            x["DollarVolume50"] = (x["Close"] * x["Volume"]).rolling(50).mean()
            result_dict[sym] = x
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
                "📊 indicators progress: {done}/{total} | elapsed: {em}m{es}s / remain: ~{rm}m{rs}s",
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


def generate_candidates_system4(
    prepared_dict: Dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    *,
    top_n: int = 10,
    progress_callback=None,
    log_callback=None,
    batch_size: int = 50,
) -> Tuple[dict, pd.DataFrame | None]:
    candidates_by_date: Dict[pd.Timestamp, list] = {}
    total = len(prepared_dict)
    start_time = time.time()
    processed, skipped = 0, 0
    buffer: list[str] = []

    spy_df = market_df.copy()
    spy_df["SMA200"] = SMAIndicator(spy_df["Close"], window=200).sma_indicator()
    spy_df["spy_filter"] = (spy_df["Close"] > spy_df["SMA200"]).astype(int)

    for sym, df in prepared_dict.items():
        try:
            x = df.copy()
            x["setup"] = (
                (x["DollarVolume50"] > 100_000_000)
                & (x["HV50"].between(10, 40))
                & (x["Close"] > x["SMA200"])
            ).astype(int)

            setup_days = x[x["setup"] == 1]
            for date, row in setup_days.iterrows():
                if date not in spy_df.index:
                    continue
                if int(spy_df.loc[date, "spy_filter"]) == 0:
                    continue
                entry_date = date + pd.Timedelta(days=1)
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
                "📊 candidates progress: {done}/{total} | elapsed: {em}m{es}s / remain: ~{rm}m{rs}s",
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


def get_total_days_system4(data_dict: Dict[str, pd.DataFrame]) -> int:
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
    "prepare_data_vectorized_system4",
    "generate_candidates_system4",
    "get_total_days_system4",
]
