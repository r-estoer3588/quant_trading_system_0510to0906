"""System1 core logic.

Provides data preparation, ROC200 ranking, and total-days helpers.
Uses precomputed indicators only - no redundant calculation.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import pandas as pd

from common.utils import (
    BatchSizeMonitor,
    describe_dtype,
    drop_duplicate_columns,
    get_cached_data,
    resolve_batch_size,
)
from common.utils_spy import resolve_signal_entry_date

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _rename_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    rename_map = {}
    for low, up in (
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ):
        if low in x.columns and up not in x.columns:
            rename_map[low] = up
    if rename_map:
        x.rename(columns=rename_map, inplace=True)
    return x


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        idx = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    else:
        idx = pd.to_datetime(df.index, errors="coerce").normalize()
    if idx is None:
        raise ValueError("invalid_date_index")
    try:
        if pd.isna(idx).all():
            raise ValueError("invalid_date_index")
    except Exception:
        pass
    x = df.copy()
    x.index = pd.Index(idx)
    x.index.name = "Date"
    x = x[~x.index.isna()]
    try:
        x = x.sort_index()
    except Exception:
        pass
    try:
        if getattr(x.index, "has_duplicates", False):
            x = x[~x.index.duplicated(keep="last")]
    except Exception:
        pass
    return x


def _prepare_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("empty_frame")
    x = _rename_ohlcv(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in x.columns]
    if missing:
        raise ValueError(f"missing_cols:{','.join(missing)}")
    x = _normalize_index(x)
    for col in REQUIRED_COLUMNS:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=[c for c in ("High", "Low", "Close") if c in x.columns])
    return x


def _compute_indicators_frame(df: pd.DataFrame) -> pd.DataFrame:
    """No-op indicator computation - check precomputed indicators exist."""
    x = df.copy()

    # Required precomputed indicators (lowercase, from indicators_common)
    required_indicators = ["sma25", "sma50", "roc200", "atr20", "dollarvolume20"]

    # Check if all required indicators exist - if not, raise error
    missing_indicators = [col for col in required_indicators if col not in x.columns]
    if missing_indicators:
        raise ValueError(f"missing precomputed indicators: {missing_indicators}")

    # Use precomputed indicators only (no calculation)
    # Create strategy-specific derived columns
    x["filter"] = (x["Low"] >= 5) & (x["dollarvolume20"] > 50_000_000)
    x["setup"] = x["filter"] & (x["sma25"] > x["sma50"])
    return x


def _compute_indicators(
    symbol: str,
    cache_dir: str,
    reuse_indicators: bool,
) -> tuple[str, pd.DataFrame | None]:
    """Check precomputed indicators for a single symbol - no computation."""
    import os

    df = get_cached_data(symbol)
    if df is None or df.empty:
        return symbol, None

    # 子プロセスから親へ簡易進捗を送る（存在すれば）
    try:
        q = globals().get("_PROGRESS_QUEUE")
        if q is not None:
            try:
                q.put((symbol, 0))
            except Exception:
                pass
    except Exception:
        pass

    # Required precomputed indicators (lowercase, from indicators_common)
    required_indicators = ["sma25", "sma50", "roc200", "atr20", "dollarvolume20"]

    # Check if all required indicators exist - immediate stop if missing
    missing_indicators = [col for col in required_indicators if col not in df.columns]
    if missing_indicators:
        raise RuntimeError(
            f"IMMEDIATE_STOP: System1 missing precomputed indicators {missing_indicators} for {symbol}. Daily signal execution must be stopped."
        )

    try:
        # Use existing processing logic but with precomputed indicators only
        x = _prepare_source_frame(df)
        x = _compute_indicators_frame(x)
    except Exception:
        return symbol, None

    # 完了を親に伝える
    try:
        q = globals().get("_PROGRESS_QUEUE")
        if q is not None:
            try:
                q.put((symbol, 100))
            except Exception:
                pass
    except Exception:
        pass

    return symbol, x


def prepare_data_vectorized_system1(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
    batch_size: int | None = None,
    reuse_indicators: bool = True,
    *,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    **kwargs,
):
    """System1 indicator computation (UI-agnostic).

    ``use_process_pool`` が True の場合、各シンボルを ProcessPoolExecutor で並列処理し、
    キャッシュを各プロセスが直接読み込む。
    ``raw_data_dict`` が None の場合は ``symbols`` からキャッシュを取得する。
    """
    import os

    # Fast-path for today-mode: if frames already include the indicators from
    # rolling/shared precompute, just ensure filter/setup exist and return.
    try:
        if reuse_indicators and isinstance(raw_data_dict, dict) and raw_data_dict:
            required = {"sma25", "sma50", "roc200", "atr20", "dollarvolume20"}
            out_fast: dict[str, pd.DataFrame] = {}
            missing: dict[str, pd.DataFrame] = {}

            for sym, df in raw_data_dict.items():
                try:
                    if df is None or df.empty:
                        missing[sym] = df
                        continue
                    x = _rename_ohlcv(df)
                    # normalize index
                    try:
                        if "Date" in x.columns:
                            x.index = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                        else:
                            x.index = pd.to_datetime(x.index, errors="coerce").normalize()
                        x = x[~x.index.isna()].sort_index()
                    except Exception:
                        pass
                    have = set(x.columns)
                    if not {"Close", "High", "Low", "Volume"}.issubset(have):
                        missing[sym] = df
                        continue
                    if not required.issubset(have):
                        raise RuntimeError(
                            f"IMMEDIATE_STOP: System1 missing precomputed indicators {required - have} for {sym}. Daily signal execution must be stopped."
                        )
                    # derive filter/setup if absent
                    if "filter" not in x.columns:
                        try:
                            x["filter"] = (x["Low"] >= 5) & (x["dollarvolume20"] > 50_000_000)
                        except Exception:
                            x["filter"] = False
                    if "setup" not in x.columns:
                        try:
                            x["setup"] = x["filter"] & (x["sma25"] > x["sma50"])
                        except Exception:
                            x["setup"] = False
                    out_fast[str(sym)] = x
                except Exception:
                    missing[str(sym)] = df

            if len(out_fast) == len(raw_data_dict):
                return out_fast
            result_dict: dict[str, pd.DataFrame] = {}
            result_dict.update(out_fast)
            if missing:
                computed = prepare_data_vectorized_system1(
                    missing,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                    skip_callback=skip_callback,
                    batch_size=batch_size,
                    reuse_indicators=False,
                    symbols=list(missing.keys()),
                    use_process_pool=use_process_pool,
                    max_workers=max_workers,
                    **kwargs,
                )
                result_dict.update(computed)
            return result_dict
    except Exception:
        pass

    cache_dir = "data_cache/indicators_system1_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # normalize inputs for both branches
    raw_data_dict = raw_data_dict or {}
    default_total_symbols = len(raw_data_dict)
    # ensure total_symbols is defined for progress callbacks in either branch
    if symbols is not None and len(symbols) > 0:
        total_symbols = len(symbols)
    else:
        total_symbols = default_total_symbols

    if use_process_pool:
        if symbols is None:
            symbols = list(raw_data_dict.keys()) if raw_data_dict else []
        total_symbols = len(symbols)
        if batch_size is None:
            try:
                from config.settings import get_settings

                batch_size = get_settings(create_dirs=False).data.batch_size
            except Exception:
                batch_size = 100
            batch_size = resolve_batch_size(total_symbols, batch_size)

        result_dict: dict[str, pd.DataFrame] = {}
        symbol_buffer: list[str] = []
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_indicators, sym, cache_dir, reuse_indicators): sym
                for sym in symbols
            }
            for i, fut in enumerate(as_completed(futures), 1):
                sym = futures[fut]
                try:
                    sym_r, df = fut.result()
                    # 念のため返却シンボル優先
                    sym = sym_r or sym
                    if df is not None:
                        result_dict[sym] = df
                        symbol_buffer.append(sym)
                except Exception as e:
                    # 1件の失敗で全体を止めない
                    if "skip_callback" in kwargs:
                        try:
                            cb = kwargs.get("skip_callback")
                            if callable(cb):
                                try:
                                    cb(sym, f"calc_error:{e}")
                                except Exception:
                                    cb(f"{sym}: calc_error:{e}")
                        except Exception:
                            pass

                if progress_callback:
                    try:
                        progress_callback(i, total_symbols)
                    except Exception:
                        pass

                # ensure batch_size is int for modulus
                try:
                    _bs = int(batch_size) if batch_size is not None else 0
                except Exception:
                    _bs = 0
                if (_bs and (i % _bs == 0 or i == total_symbols)) and log_callback:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / i) * (total_symbols - i) if i else 0
                    em, es = divmod(int(elapsed), 60)
                    rm, rs = divmod(int(remaining), 60)
                    joined_syms = ", ".join(symbol_buffer)
                    try:
                        # split into multiple shorter calls to avoid long single-line strings
                        log_callback(
                            f"📊 System1 indicators: {i}/{total_symbols} | "
                            f"elapsed: {em}m{es}s / remain: ~{rm}m{rs}s"
                        )
                        if joined_syms:
                            log_callback(f"symbols: {joined_syms}")
                    except Exception:
                        pass
                    symbol_buffer.clear()

        return result_dict

    # Regular mode: process in main thread
    result_dict: dict[str, pd.DataFrame] = {}
    total_symbols = len(raw_data_dict)

    for i, (sym, df) in enumerate(raw_data_dict.items(), 1):
        try:
            if df is None or df.empty:
                if skip_callback:
                    skip_callback(sym, "empty_dataframe")
                continue

            # Required precomputed indicators (lowercase, from indicators_common)
            required_indicators = ["sma25", "sma50", "roc200", "atr20", "dollarvolume20"]

            # Check if all required indicators exist
            missing_indicators = [col for col in required_indicators if col not in df.columns]
            if missing_indicators:
                if skip_callback:
                    skip_callback(sym, f"missing_precomputed:{','.join(missing_indicators)}")
                continue

            # Use only precomputed indicators - no calculation
            x = _prepare_source_frame(df)
            x = _compute_indicators_frame(x)
            result_dict[sym] = x

        except Exception as e:
            if skip_callback:
                skip_callback(sym, f"processing_error:{e}")

        if progress_callback:
            try:
                progress_callback(i, total_symbols)
            except Exception:
                pass

    return result_dict


def generate_roc200_ranking_system1(
    prepared_data: dict[str, pd.DataFrame],
    entry_date: pd.Timestamp,
    top_n: int = 20,
    *,
    use_today: bool = False,
    progress_callback=None,
    **kwargs,
) -> pd.DataFrame:
    """Generate ROC200 ranking from prepared data."""
    candidates = []
    total = len(prepared_data)

    for i, (symbol, df) in enumerate(prepared_data.items(), 1):
        try:
            if df is None or df.empty:
                continue

            # Use lowercase column names consistently
            if "roc200" not in df.columns:
                continue

            # Get data for entry date
            available_dates = df.index[df.index <= entry_date]
            if available_dates.empty:
                continue

            target_date = available_dates[-1]
            row = df.loc[target_date]

            # Check setup conditions
            if not row.get("setup", False):
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "entry_date": target_date,
                    "roc200": row["roc200"],
                    "setup": True,
                }
            )

        except Exception:
            continue

        if progress_callback and i % 100 == 0:
            try:
                progress_callback(i, total)
            except Exception:
                pass

    if not candidates:
        return pd.DataFrame()

    # Convert to DataFrame and sort by ROC200 descending
    result_df = pd.DataFrame(candidates)
    result_df = result_df.sort_values("roc200", ascending=False).head(top_n)

    return result_df


def total_days_for_system1() -> int:
    """Days required for all System1 indicators to stabilize."""
    return max(25, 50, 200, 20)  # SMA25, SMA50, ROC200, ATR20


def get_total_days_system1(data_dict: dict[str, pd.DataFrame]) -> int:
    """Return the total number of unique dates across prepared data."""
    all_dates = set()
    for df in data_dict.values():
        if df is None or df.empty:
            continue
        if "Date" in df.columns:
            date_series = pd.to_datetime(df["Date"]).dt.normalize()
        else:
            date_series = pd.to_datetime(df.index).normalize()
        all_dates.update(date_series)
    return len(sorted(all_dates))


def generate_roc200_ranking_system1(data_dict: dict, spy_df: pd.DataFrame, **kwargs):
    """Generate daily ROC200 ranking filtered by SPY trend."""
    from common.utils_spy import resolve_signal_entry_date

    on_progress = kwargs.get("on_progress")
    on_log = kwargs.get("on_log")
    all_signals = []

    for symbol, df in data_dict.items():
        if "setup" not in df.columns or df["setup"].sum() == 0:
            continue
        sig_df = df[df["setup"]][["ROC200", "ATR20", "Open"]].copy()
        sig_df["symbol"] = symbol
        idx_norm = pd.to_datetime(sig_df.index, errors="coerce").normalize()
        sig_df["Date"] = idx_norm

        # last_price（直近終値）を取得
        last_price = None
        if "Close" in df.columns and not df["Close"].empty:
            last_price = df["Close"].iloc[-1]
        sig_df["entry_price"] = last_price

        # entry_date 計算
        base_dates = pd.to_datetime(sig_df.index, errors="coerce").to_series(index=sig_df.index)
        sig_df["entry_date"] = base_dates.map(resolve_signal_entry_date)
        sig_df = sig_df.dropna(subset=["entry_date"])

        all_signals.append(sig_df)

    if not all_signals:
        return {}, None

    all_df = pd.concat(all_signals)

    # entry_date単位でランキング作成
    candidates_by_date: dict[pd.Timestamp, list[dict]] = {}
    for date, group in all_df.groupby("entry_date"):
        ranked = group.sort_values("ROC200", ascending=False).copy()
        total = len(ranked)
        if total == 0:
            candidates_by_date[date] = []
            continue
        ranked.loc[:, "rank"] = range(1, total + 1)
        ranked.loc[:, "rank_total"] = total
        candidates_by_date[date] = ranked.to_dict("records")

    return candidates_by_date, None


__all__ = [
    "prepare_data_vectorized_system1",
    "generate_roc200_ranking_system1",
    "get_total_days_system1",
    "total_days_for_system1",
]
