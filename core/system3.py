"""System3 core logic (Long mean-reversion)."""

from typing import Dict, Tuple
import time
import pandas as pd
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange
from common.i18n import tr


def prepare_data_vectorized_system3(
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
    buffer = []

    for sym, df in raw_data_dict.items():
        x = df.copy()
        if len(x) < 150:
            skipped += 1
            processed += 1
            continue

        try:
            x["SMA150"] = SMAIndicator(x["Close"], window=150).sma_indicator()
            x["ATR10"] = AverageTrueRange(x["High"], x["Low"], x["Close"], window=10).average_true_range()
            x["DropRate_3D"] = -(x["Close"].pct_change(3))
            x["AvgVolume50"] = x["Volume"].rolling(50).mean()
            x["ATR_Ratio"] = x["ATR10"] / x["Close"]

            x["setup"] = (
                (x["Close"] > x["SMA150"]) & (x["DropRate_3D"] >= 0.125)
                & (x["Close"] > 1) & (x["AvgVolume50"] >= 1_000_000)
                & (x["ATR_Ratio"] >= 0.05)
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


def generate_candidates_system3(
    prepared_dict: Dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
    progress_callback=None,
    log_callback=None,
    batch_size: int = 50,
) -> Tuple[dict, pd.DataFrame | None]:
    all_signals = []
    total = len(prepared_dict)
    processed = 0
    buffer = []
    start_time = time.time()

    for sym, df in prepared_dict.items():
        if "setup" not in df.columns or not df["setup"].any():
            continue
        setup_df = df[df["setup"] == 1].copy()
        setup_df["symbol"] = sym
        setup_df["entry_date"] = setup_df.index + pd.Timedelta(days=1)
        setup_df = setup_df[["symbol", "entry_date", "DropRate_3D", "ATR10"]]
        all_signals.append(setup_df)
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

    if not all_signals:
        return {}, None

    all_df = pd.concat(all_signals)
    candidates_by_date = {}
    for date, group in all_df.groupby("entry_date"):
        ranked = group.sort_values("DropRate_3D", ascending=False)
        candidates_by_date[date] = ranked.head(int(top_n)).to_dict("records")
    return candidates_by_date, None


def get_total_days_system3(data_dict: Dict[str, pd.DataFrame]) -> int:
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
    "prepare_data_vectorized_system3",
    "generate_candidates_system3",
    "get_total_days_system3",
]
