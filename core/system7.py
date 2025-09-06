"""System7 core logic (SPY short catastrophe hedge)。

System7 は SPY 専用のため、prepare_data/generate_candidates のみ共通化。
run_backtest は strategy 側にカスタム実装が残る。
"""

from typing import Dict, Tuple
import pandas as pd
from ta.volatility import AverageTrueRange


def prepare_data_vectorized_system7(
    raw_data_dict: Dict[str, pd.DataFrame],
    *,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
) -> Dict[str, pd.DataFrame]:
    prepared_dict: Dict[str, pd.DataFrame] = {}
    try:
        df = raw_data_dict.get("SPY").copy()
        df["ATR50"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=50).average_true_range()
        df["min_50"] = df["Close"].rolling(window=50).min().round(4)
        df["setup"] = (df["Low"] <= df["min_50"]).astype(int)
        df["max_70"] = df["Close"].rolling(window=70).max()
        prepared_dict["SPY"] = df
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
    prepared_dict: Dict[str, pd.DataFrame],
    *,
    progress_callback=None,
    log_callback=None,
) -> Tuple[dict, pd.DataFrame | None]:
    candidates_by_date: Dict[pd.Timestamp, list] = {}
    if "SPY" not in prepared_dict:
        return {}, None
    df = prepared_dict["SPY"]
    setup_days = df[df["setup"] == 1]
    for date, row in setup_days.iterrows():
        entry_idx = df.index.get_loc(date)
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


def get_total_days_system7(data_dict: Dict[str, pd.DataFrame]) -> int:
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
