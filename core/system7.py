"""System7 core logic (SPY short catastrophe hedge)。

System7 は SPY 専用のため、prepare_data/generate_candidates のみ共通化。
run_backtest は strategy 側にカスタム実装が残る。
"""

import os

import pandas as pd
from ta.volatility import AverageTrueRange


def prepare_data_vectorized_system7(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    *,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
    reuse_indicators: bool = True,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Compute indicators for SPY and cache the result."""
    cache_dir = "data_cache/indicators_system7_cache"
    os.makedirs(cache_dir, exist_ok=True)
    prepared_dict: dict[str, pd.DataFrame] = {}
    raw_data_dict = raw_data_dict or {}
    try:
        df_raw = raw_data_dict.get("SPY")
        if df_raw is None:
            raise ValueError("SPY data missing")
        if "Date" in df_raw.columns:
            df = df_raw.copy()
            df.index = pd.Index(pd.to_datetime(df["Date"]).dt.normalize())
        else:
            df = df_raw.copy()
            df.index = pd.Index(pd.to_datetime(df.index).normalize())

        cache_path = os.path.join(cache_dir, "SPY.feather")
        cached: pd.DataFrame | None = None
        if reuse_indicators and os.path.exists(cache_path):
            try:
                cached = pd.read_feather(cache_path)
                cached["Date"] = pd.to_datetime(cached["Date"]).dt.normalize()
                cached.set_index("Date", inplace=True)
            except Exception:
                cached = None

        def _calc_indicators(src: pd.DataFrame) -> pd.DataFrame:
            x = src.copy()
            x["ATR50"] = AverageTrueRange(
                x["High"], x["Low"], x["Close"], window=50
            ).average_true_range()
            x["min_50"] = x["Low"].rolling(50).min().round(4)
            x["setup"] = (x["Low"] <= x["min_50"]).astype(int)
            x["max_70"] = x["Close"].rolling(70).max()
            return x

        if cached is not None and not cached.empty:
            last_date = cached.index.max()
            new_rows = df[df.index > last_date]
            if new_rows.empty:
                result_df = cached
            else:
                context_start = last_date - pd.Timedelta(days=70)
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
        prepared_dict["SPY"] = result_df
    except Exception as e:
        if skip_callback:
            try:
                skip_callback(f"SPY の処理をスキップしました: {e}")
            except Exception:
                pass

    if log_callback:
        try:
            log_callback("SPY インジケーター計算完了(ATR50, min_50, max_70, setup)")
        except Exception:
            pass
    if progress_callback:
        try:
            progress_callback(1, 1)
        except Exception:
            pass
    return prepared_dict


def generate_candidates_system7(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    progress_callback=None,
    log_callback=None,
) -> tuple[dict, pd.DataFrame | None]:
    candidates_by_date: dict[pd.Timestamp, list] = {}
    if "SPY" not in prepared_dict:
        return {}, None
    df = prepared_dict["SPY"]
    setup_days = df[df["setup"] == 1]
    for date, row in setup_days.iterrows():
        entry_idx = int(df.index.get_loc(date))
        if entry_idx + 1 >= len(df):
            continue
        entry_date = df.index[entry_idx + 1]
        rec = {"symbol": "SPY", "entry_date": entry_date, "ATR50": row["ATR50"]}
        candidates_by_date.setdefault(entry_date, []).append(rec)
    if log_callback:
        try:
            log_callback(f"候補日数: {len(candidates_by_date)}")
        except Exception:
            pass
    if progress_callback:
        try:
            progress_callback(1, 1)
        except Exception:
            pass
    return candidates_by_date, None


def get_total_days_system7(data_dict: dict[str, pd.DataFrame]) -> int:
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
    "prepare_data_vectorized_system7",
    "generate_candidates_system7",
    "get_total_days_system7",
]
