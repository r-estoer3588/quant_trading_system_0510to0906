"""System7 core logic (SPY short catastrophe hedge)。

System7 は SPY 専用のため、prepare_data/generate_candidates のみ共通化。
run_backtest は strategy 側にカスタム実装が残る。
"""

import os

import pandas as pd
from ta.volatility import AverageTrueRange

from common.utils_spy import resolve_signal_entry_date


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
        use_cache = bool(reuse_indicators and len(df) >= 300)
        cached: pd.DataFrame | None = None
        if use_cache and os.path.exists(cache_path):
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
            # max_70 は既存値を尊重（全行埋まっていれば再計算しない）
            if "max_70" not in x.columns:
                x["max_70"] = x["Close"].rolling(70).max()
            else:
                # 既存カラムが部分的に NaN の場合は、NaN のみを埋める
                need = ~x["max_70"].notna()
                if need.any():
                    x.loc[need, "max_70"] = x["Close"].rolling(70).max()[need]
            return x

        if use_cache and cached is not None and not cached.empty:
            last_date = cached.index.max()
            new_rows = df[df.index > last_date]
            if new_rows.empty:
                result_df = cached
            else:
                context_start = last_date - pd.Timedelta(days=70)
                recompute_src = df[df.index >= context_start]
                recomputed = _calc_indicators(recompute_src)
                recomputed = recomputed[recomputed.index > last_date]
                # 既存の max_70 を優先して結合
                result_df = pd.concat([cached, recomputed])
                if "max_70" in cached.columns and "max_70" in recomputed.columns:
                    # cached 側の値を優先（重複期間は cached を保持）
                    result_df.loc[cached.index, "max_70"] = cached["max_70"]
                try:
                    result_df.reset_index().to_feather(cache_path)
                except Exception:
                    pass
        else:
            result_df = _calc_indicators(df)
            try:
                if use_cache:
                    result_df.reset_index().to_feather(cache_path)
            except Exception:
                pass
        # 原データに max_70 が含まれている場合はそれを最優先で反映（テスト互換のため）
        if "max_70" in df.columns and not df["max_70"].isna().all():
            common_idx = df.index.intersection(result_df.index)
            if len(common_idx) > 0:
                result_df.loc[common_idx, "max_70"] = df.loc[common_idx, "max_70"]
        # テスト互換: 返却範囲は入力 df のインデックスに厳密一致させる
        try:
            result_df = result_df.reindex(df.index)
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
            log_callback("SPY インジケーター計算完了(ATR50, min_50, max_70, setup: Low<=min_50)")
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
        entry_date = resolve_signal_entry_date(date)
        if pd.isna(entry_date):
            continue
        # last_price（直近終値）を取得
        last_price = None
        if "Close" in df.columns and not df["Close"].empty:
            last_price = df["Close"].iloc[-1]
        rec = {
            "symbol": "SPY",
            "entry_date": entry_date,
            "ATR50": row["ATR50"],
            "entry_price": last_price,
        }
        candidates_by_date.setdefault(entry_date, []).append(rec)
    if log_callback:
        try:
            # 直近のセットアップ（50日安値ブレイク）に基づく、
            # 翌営業日のユニークなエントリー予定日数を、過去50営業日に限定して集計
            all_dates = pd.Index(pd.to_datetime(df.index).normalize()).unique().sort_values()
            window_size = int(min(50, len(all_dates)) or 50)
            if window_size > 0:
                recent_set = set(all_dates[-window_size:])
            else:
                recent_set = set()
            count_50 = sum(1 for d in candidates_by_date.keys() if d in recent_set)
            log_callback(
                f"候補日数: {count_50} "
                f"(直近({count_50}/{window_size})日間, 50日安値由来の翌営業日数)"
            )
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
