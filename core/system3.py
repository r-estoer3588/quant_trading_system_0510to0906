"""System3 core logic (Long mean-reversion)."""

import time

import pandas as pd
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange

from common.i18n import tr
from common.utils import resolve_batch_size

# Trading thresholds - Default values for business rules
DEFAULT_ATR_RATIO_THRESHOLD = 0.05  # 5% ATR ratio threshold for filtering


def prepare_data_vectorized_system3(
    raw_data_dict: dict[str, pd.DataFrame],
    *,
    progress_callback=None,
    log_callback=None,
    batch_size: int | None = None,
) -> dict[str, pd.DataFrame]:
    result_dict: dict[str, pd.DataFrame] = {}
    total = len(raw_data_dict)
    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        batch_size = resolve_batch_size(total, batch_size)
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
        setup_df["entry_date"] = setup_df.index + pd.Timedelta(days=1)
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
        else:
            dates = pd.to_datetime(df.index).normalize()
        all_dates.update(dates)
    return len(all_dates)


__all__ = [
    "prepare_data_vectorized_system3",
    "generate_candidates_system3",
    "get_total_days_system3",
]
