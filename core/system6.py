"""System6 core logic (Short mean-reversion momentum burst)."""

import time

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from ta.volatility import AverageTrueRange

from common.i18n import tr
from common.utils import get_cached_data, resolve_batch_size


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    df = get_cached_data(symbol)
    if df is None or df.empty:
        return symbol, None
    x = df.copy()
    if len(x) < 50:
        return symbol, None
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
    except Exception:
        return symbol, None
    return symbol, x


def prepare_data_vectorized_system6(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    *,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
    batch_size: int | None = None,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
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
                            skip_callback(f"{sym} スキップ")
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
    processed, skipped = 0, 0
    buffer: list[str] = []

    for sym, df in raw_data_dict.items():
        x = df.copy()
        if len(x) < 50:
            skipped += 1
            processed += 1
            continue
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

    if skipped > 0:
        msg = f"⚠️ データ不足/計算失敗でスキップ: {skipped} 件"
        try:
            if skip_callback:
                skip_callback(msg)
            elif log_callback:
                log_callback(msg)
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
    buffer: list[str] = []

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
                entry_date = date + pd.Timedelta(days=1)
                if entry_date not in df.index:
                    continue
                rec = {
                    "symbol": sym,
                    "entry_date": entry_date,
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
                msg += "\n" + tr("symbols: {names}", names=", ".join(buffer))
            try:
                log_callback(msg)
            except Exception:
                pass
            buffer.clear()

    for date in list(candidates_by_date.keys()):
        ranked = sorted(
            candidates_by_date[date],
            key=lambda r: r["Return6D"],
            reverse=True,
        )
        candidates_by_date[date] = ranked[: int(top_n)]

    if skipped > 0:
        msg = f"⚠️ 候補抽出中にスキップ: {skipped} 件"
        try:
            if skip_callback:
                skip_callback(msg)
            elif log_callback:
                log_callback(msg)
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
        else:
            dates = pd.to_datetime(df.index).normalize()
        all_dates.update(dates)
    return len(all_dates)


__all__ = [
    "prepare_data_vectorized_system6",
    "generate_candidates_system6",
    "get_total_days_system6",
]
