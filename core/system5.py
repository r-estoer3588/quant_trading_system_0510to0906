"""System5 core logic (Long mean-reversion with high ADX)."""

from typing import Dict, Tuple
import time
import pandas as pd
from ta.trend import SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from common.i18n import tr


def prepare_data_vectorized_system5(
    raw_data_dict: Dict[str, pd.DataFrame],
    *,
    progress_callback=None,
    log_callback=None,
    batch_size: int = 50,
) -> Dict[str, pd.DataFrame]:
    result_dict: Dict[str, pd.DataFrame] = {}
    total = len(raw_data_dict)
    processed, skipped = 0, 0
    buffer: list[str] = []
    start_time = time.time()

    for sym, df in raw_data_dict.items():
        x = df.copy()
        if len(x) < 100:
            skipped += 1
            processed += 1
            continue
        try:
            x["SMA100"] = SMAIndicator(x["Close"], window=100).sma_indicator()
            x["ATR10"] = AverageTrueRange(x["High"], x["Low"], x["Close"], window=10).average_true_range()
            x["ADX7"] = ADXIndicator(x["High"], x["Low"], x["Close"], window=7).adx()
            x["RSI3"] = RSIIndicator(x["Close"], window=3).rsi()
            x["AvgVolume50"] = x["Volume"].rolling(50).mean()
            x["DollarVolume50"] = (x["Close"] * x["Volume"]).rolling(50).mean()
            x["ATR_Pct"] = x["ATR10"] / x["Close"]

            x["setup"] = (
                (x["Close"] > x["SMA100"] + x["ATR10"]) & (x["ADX7"] > 55)
                & (x["RSI3"] < 50) & (x["AvgVolume50"] > 500_000)
                & (x["DollarVolume50"] > 2_500_000) & (x["ATR_Pct"] > 0.04)
            ).astype(int)

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


def generate_candidates_system5(
    prepared_dict: Dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
    progress_callback=None,
    log_callback=None,
    batch_size: int = 50,
) -> Tuple[dict, pd.DataFrame | None]:
    candidates_by_date: Dict[pd.Timestamp, list] = {}
    total = len(prepared_dict)
    processed, skipped = 0, 0
    buffer: list[str] = []
    start_time = time.time()

    for sym, df in prepared_dict.items():
        try:
            setup_days = df[df["setup"] == 1]
            for date, row in setup_days.iterrows():
                entry_date = date + pd.Timedelta(days=1)
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

    for date in list(candidates_by_date.keys()):
        ranked = sorted(candidates_by_date[date], key=lambda r: r["ADX7"], reverse=True)
        candidates_by_date[date] = ranked[: int(top_n)]

    if skipped > 0 and log_callback:
        try:
            log_callback(f"⚠️ 候補抽出中にスキップ: {skipped} 件")
        except Exception:
            pass
    return candidates_by_date, None


def get_total_days_system5(data_dict: Dict[str, pd.DataFrame]) -> int:
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
