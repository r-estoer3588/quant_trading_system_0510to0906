from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time as _t
from typing import Any, cast
import numpy as np

import pandas as pd

from config.settings import get_settings
from core.system5 import (
    DEFAULT_ATR_PCT_THRESHOLD,
    format_atr_pct_threshold_label,
)
from common.utils_spy import get_next_nyse_trading_day, get_spy_with_indicators

SpyGate = Callable[[pd.Timestamp], bool]

# --- サイド定義（売買区分）---
# System1/3/5 は買い戦略、System2/4/6/7 は売り戦略として扱う。
LONG_SYSTEMS = {"system1", "system3", "system5"}
SHORT_SYSTEMS = {"system2", "system4", "system6", "system7"}


@dataclass(frozen=True)
class TodaySignal:
    symbol: str
    system: str
    side: str  # "long" | "short"
    signal_type: str  # "buy" | "sell"
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    score_key: str | None = None
    score: float | None = None
    reason: str | None = None


def _empty_signals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "system",
            "side",
            "signal_type",
            "entry_date",
            "entry_price",
            "stop_price",
            "score_key",
            "score",
        ]
    )


def _infer_side(system_name: str) -> str:
    name = (system_name or "").lower()
    if name in SHORT_SYSTEMS:
        return "short"
    return "long"


def _score_from_candidate(
    system_name: str, candidate: dict
) -> tuple[str | None, float | None, bool]:
    """
    候補レコードからスコア項目と並び順（昇順か）を推定して返す。
    戻り値: (score_key, score_value, asc)
    """
    name = (system_name or "").lower()
    # System7 は SPY 専用ヘッジ。ATR50 はストップ計算用のため、
    # スコア/理由には使用しない（スコア欄は空にする）。
    if name == "system7":
        return None, None, False
    # システム別の代表スコア
    key_order: list[tuple[list[str], bool]] = [
        (["ROC200"], False),  # s1: 大きいほど良い
        (["ADX7"], False),  # s2,s5: 大きいほど良い
        (["Drop3D"], False),  # s3: 大きいほど良い（下落率）
        (["RSI4"], True),  # s4: 小さいほど良い
        (["Return6D"], False),  # s6: 大きいほど良い
        (["ATR50"], False),  # s7: 参考
    ]
    # system 固有優先順位
    if name == "system4":
        key_order = [(["RSI4"], True), (["ATR40"], True)] + key_order
    elif name == "system2":
        key_order = [(["ADX7"], False), (["RSI3"], False)] + key_order
    elif name == "system5":
        key_order = [(["ADX7"], False), (["ATR10"], True)] + key_order
    elif name == "system6":
        key_order = [(["Return6D"], False), (["ATR10"], True)] + key_order

    for keys, asc in key_order:
        for k in keys:
            if k in candidate:
                v = candidate.get(k)
                if v is None:
                    return k, None, asc
                if isinstance(v, (int, float, str)):
                    try:
                        return k, float(v), asc
                    except Exception:
                        return k, None, asc
                else:
                    return k, None, asc
    # 見つからない場合
    return None, None, False


def _label_for_score_key(key: str | None) -> str:
    """スコアキーの日本語ラベルを返す（既知のもののみ簡潔表示）。"""
    if key is None:
        return "スコア"
    k = str(key).upper()
    mapping = {
        "ROC200": "ROC200",
        "ADX7": "ADX",
        "RSI4": "RSI4",
        "RSI3": "RSI3",
        "DROP3D": "3日下落率",
        "RETURN6D": "過去6日騰落率",
        "ATR10": "ATR10",
        "ATR20": "ATR20",
        "ATR40": "ATR40",
        "ATR50": "ATR50",
    }
    return mapping.get(k, k)


def _asc_by_score_key(score_key: str | None) -> bool:
    """スコアキーごとの昇順/降順を判定。"""
    return bool(score_key and score_key.upper() in {"RSI4"})


def _pick_atr_col(df: pd.DataFrame) -> str | None:
    for col in ("ATR20", "ATR10", "ATR40", "ATR50", "ATR14"):
        if col in df.columns:
            return col
    return None


def _compute_entry_stop(
    strategy, df: pd.DataFrame, candidate: dict, side: str
) -> tuple[float, float] | None:
    # strategy 独自の compute_entry があれば優先
    try:
        _fn = strategy.compute_entry  # type: ignore[attr-defined]
    except Exception:
        _fn = None
    if callable(_fn):
        try:
            res = _fn(df, candidate, 0.0)
            if res and isinstance(res, tuple) and len(res) == 2:
                entry, stop = float(res[0]), float(res[1])
                if entry > 0 and (
                    (side == "short" and stop > entry)
                    or (side == "long" and entry > stop)
                ):
                    return round(entry, 4), round(stop, 4)
        except Exception:
            pass

    # フォールバック: 当日始値 ± 3*ATR
    try:
        entry_ts = pd.Timestamp(candidate["entry_date"])
    except Exception:
        return None
    try:
        idxer = df.index.get_indexer([entry_ts])
        entry_idx = int(idxer[0]) if len(idxer) else -1
    except Exception:
        return None
    if entry_idx <= 0 or entry_idx >= len(df):
        return None
    try:
        entry = float(df.iloc[entry_idx]["Open"])
    except Exception:
        return None
    atr_col = _pick_atr_col(df)
    if not atr_col:
        return None
    try:
        atr = float(df.iloc[entry_idx - 1][atr_col])
    except Exception:
        return None
    mult = 3.0
    stop = entry - mult * atr if side == "long" else entry + mult * atr
    return round(entry, 4), round(stop, 4)


def _normalize_symbol_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or getattr(df, "empty", True):
        return None
    try:
        x = df.copy()
        raw_dates = (
            pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
            if "Date" in x.columns
            else pd.to_datetime(x.index, errors="coerce").normalize()
        )
        date_series = pd.Series(raw_dates, index=x.index)
        mask = ~date_series.isna()
        if not mask.any():
            return None
        x = x.loc[mask].copy()
        date_index = pd.DatetimeIndex(date_series.loc[mask])
        if date_index.empty:
            return None
        x.index = date_index
        x = x.sort_index()
        if getattr(x.index, "has_duplicates", False):
            x = x[~x.index.duplicated(keep="last")]
        return x
    except Exception:
        return None


def _row_on_or_before(df: pd.DataFrame, target: pd.Timestamp) -> pd.Series | None:
    if df is None or getattr(df, "empty", True):
        return None
    try:
        if target in df.index:
            row = df.loc[target]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            return row
    except Exception:
        pass
    try:
        mask_arr = np.asarray(df.index <= target)
        if mask_arr.any():
            idx = int(np.nonzero(mask_arr)[0][-1])
            return df.iloc[idx]
    except Exception:
        pass
    try:
        return df.iloc[-1]
    except Exception:
        return None


def _build_latest_rows(
    prepared: dict[str, pd.DataFrame], prev_day: pd.Timestamp
) -> tuple[dict[str, pd.Series], int]:
    latest_rows: dict[str, pd.Series] = {}
    filter_pass = 0
    for sym, df in prepared.items():
        row = _row_on_or_before(df, prev_day)
        if row is None:
            continue
        latest_rows[sym] = row
        try:
            if bool(row.get("filter")):
                filter_pass += 1
        except Exception:
            continue
    return latest_rows, filter_pass


def _make_spy_gate(spy_df: pd.DataFrame | None, column: str) -> SpyGate:
    if spy_df is None or getattr(spy_df, "empty", True) or column not in spy_df.columns:
        return lambda _: True

    def _gate(ts: pd.Timestamp) -> bool:
        row = _row_on_or_before(spy_df, ts)
        if row is None:
            return True
        try:
            close_val = float(row.get("Close", float("nan")))
            sma_val = float(row.get(column, float("nan")))
            if np.isnan(close_val) or np.isnan(sma_val):
                return True
            return close_val > sma_val
        except Exception:
            return True

    return _gate


def _compute_setup_pass(
    system_name: str,
    latest_rows: dict[str, pd.Series],
    filter_pass: int,
    prev_day: pd.Timestamp,
    log_callback: Callable[[str], None] | None,
    spy_df: pd.DataFrame | None,
) -> int:
    rows_list = list(latest_rows.values())
    name = system_name.lower()
    setup_pass = 0

    def _safe_float(val: Any) -> float:
        try:
            num = float(val)
            if np.isnan(num):
                raise ValueError
            return num
        except Exception:
            return float("nan")

    if name == "system1":
        filtered_rows = [r for r in rows_list if bool(r.get("filter"))]

        def _sma_ok(row: pd.Series) -> bool:
            left = _safe_float(row.get("SMA25"))
            right = _safe_float(row.get("SMA50"))
            if np.isnan(left) or np.isnan(right):
                return False
            return left > right

        sma_pass = sum(1 for row in filtered_rows if _sma_ok(row))
        spy_gate: int | None = None
        if spy_df is not None and not getattr(spy_df, "empty", True):
            spy_row = _row_on_or_before(spy_df, prev_day)
            if spy_row is not None:
                close_val = _safe_float(spy_row.get("Close"))
                sma_val = _safe_float(spy_row.get("SMA100"))
                if not np.isnan(close_val) and not np.isnan(sma_val):
                    spy_gate = 1 if close_val > sma_val else 0
        setup_pass = sma_pass if spy_gate != 0 else 0
        if log_callback:
            spy_label = "-" if spy_gate is None else str(int(spy_gate))
            try:
                log_callback(
                    "🧩 system1セットアップ内訳: "
                    + f"フィルタ通過={filter_pass}, SPY>SMA100: {spy_label}, "
                    + f"SMA25>SMA50: {sma_pass}"
                )
            except Exception:
                pass
        return int(setup_pass)

    if name == "system2":
        filtered_rows = [r for r in rows_list if bool(r.get("filter"))]

        def _rsi_ok(row: pd.Series) -> bool:
            val = _safe_float(row.get("RSI3"))
            return not np.isnan(val) and val > 90

        def _two_up(row: pd.Series) -> bool:
            try:
                return bool(row.get("TwoDayUp"))
            except Exception:
                return False

        rsi_pass = sum(1 for row in filtered_rows if _rsi_ok(row))
        two_up_pass = sum(1 for row in filtered_rows if _rsi_ok(row) and _two_up(row))
        setup_pass = two_up_pass
        if log_callback:
            try:
                log_callback(
                    "🧩 system2セットアップ内訳: "
                    + f"フィルタ通過={filter_pass}, RSI3>90: {rsi_pass}, "
                    + f"TwoDayUp: {two_up_pass}"
                )
            except Exception:
                pass
        return int(setup_pass)

    if name == "system3":
        filtered_rows = [r for r in rows_list if bool(r.get("filter"))]

        def _close_ok(row: pd.Series) -> bool:
            close_val = _safe_float(row.get("Close"))
            sma_val = _safe_float(row.get("SMA150"))
            if np.isnan(close_val) or np.isnan(sma_val):
                return False
            return close_val > sma_val

        def _drop_ok(row: pd.Series) -> bool:
            drop_val = _safe_float(row.get("Drop3D"))
            return not np.isnan(drop_val) and drop_val >= 0.125

        close_pass = sum(1 for row in filtered_rows if _close_ok(row))
        drop_pass = sum(1 for row in filtered_rows if _close_ok(row) and _drop_ok(row))
        setup_pass = drop_pass
        if log_callback:
            try:
                log_callback(
                    "🧩 system3セットアップ内訳: "
                    + f"フィルタ通過={filter_pass}, Close>SMA150: {close_pass}, "
                    + f"3日下落率>=12.5%: {drop_pass}"
                )
            except Exception:
                pass
        return int(setup_pass)

    if name == "system4":
        def _above_sma(row: pd.Series) -> bool:
            if not bool(row.get("filter")):
                return False
            close_val = _safe_float(row.get("Close"))
            sma_val = _safe_float(row.get("SMA200"))
            if np.isnan(close_val) or np.isnan(sma_val):
                return False
            return close_val > sma_val

        above_sma = sum(1 for row in rows_list if _above_sma(row))
        setup_pass = above_sma
        if log_callback:
            try:
                log_callback(
                    "🧩 system4セットアップ内訳: "
                    + f"フィルタ通過={filter_pass}, Close>SMA200: {above_sma}"
                )
            except Exception:
                pass
        return int(setup_pass)

    if name == "system5":
        threshold_label = format_atr_pct_threshold_label()
        rows_total = len(rows_list)
        s5_av = 0
        s5_dv = 0
        s5_atr = 0
        for row in rows_list:
            av_val = _safe_float(row.get("AvgVolume50"))
            if np.isnan(av_val) or av_val <= 500_000:
                continue
            s5_av += 1
            dv_val = _safe_float(row.get("DollarVolume50"))
            if np.isnan(dv_val) or dv_val <= 2_500_000:
                continue
            s5_dv += 1
            atr_pct_val = _safe_float(row.get("ATR_Pct"))
            if not np.isnan(atr_pct_val) and atr_pct_val > DEFAULT_ATR_PCT_THRESHOLD:
                s5_atr += 1
        if log_callback:
            try:
                log_callback(
                    "🧪 system5内訳: "
                    + f"対象={rows_total}, AvgVol50>500k: {s5_av}, "
                    + f"DV50>2.5M: {s5_dv}, {threshold_label}: {s5_atr}"
                )
            except Exception:
                pass

        def _price_ok(row: pd.Series) -> bool:
            if not bool(row.get("filter")):
                return False
            close_val = _safe_float(row.get("Close"))
            sma_val = _safe_float(row.get("SMA100"))
            atr_val = _safe_float(row.get("ATR10"))
            if any(np.isnan(v) for v in (close_val, sma_val, atr_val)):
                return False
            return close_val > sma_val + atr_val

        def _adx_ok(row: pd.Series) -> bool:
            val = _safe_float(row.get("ADX7"))
            return not np.isnan(val) and val > 55

        def _rsi_ok(row: pd.Series) -> bool:
            val = _safe_float(row.get("RSI3"))
            return not np.isnan(val) and val < 50

        price_pass = sum(1 for row in rows_list if _price_ok(row))
        adx_pass = sum(1 for row in rows_list if _price_ok(row) and _adx_ok(row))
        rsi_pass = sum(
            1 for row in rows_list if _price_ok(row) and _adx_ok(row) and _rsi_ok(row)
        )
        setup_pass = rsi_pass
        if log_callback:
            try:
                log_callback(
                    "🧩 system5セットアップ内訳: "
                    + f"フィルタ通過={filter_pass}, Close>SMA100+ATR10: {price_pass}, "
                    + f"ADX7>55: {adx_pass}, RSI3<50: {rsi_pass}"
                )
            except Exception:
                pass
        return int(setup_pass)

    if name == "system6":
        filtered_rows = [r for r in rows_list if bool(r.get("filter"))]

        def _ret_ok(row: pd.Series) -> bool:
            val = _safe_float(row.get("Return6D"))
            return not np.isnan(val) and val > 0.20

        def _up_two(row: pd.Series) -> bool:
            try:
                return bool(row.get("UpTwoDays"))
            except Exception:
                return False

        ret_pass = sum(1 for row in filtered_rows if _ret_ok(row))
        up_pass = sum(1 for row in filtered_rows if _ret_ok(row) and _up_two(row))
        setup_pass = up_pass
        if log_callback:
            try:
                msg = (
                    "🧩 system6セットアップ内訳: "
                    f"フィルタ通過={filter_pass}, "
                    f"Return6D>20%: {ret_pass}, "
                    f"UpTwoDays: {up_pass}"
                )
                log_callback(msg)
            except Exception:
                pass
        return int(setup_pass)

    if name == "system7":
        spy_present = 1 if "SPY" in latest_rows else 0
        setup_pass = spy_present
        if log_callback:
            try:
                msg = f"🧩 system7セットアップ内訳: SPY存在={spy_present}"
                if spy_present:
                    try:
                        spy_row = latest_rows.get("SPY", pd.Series())
                        setup_raw = spy_row.get("setup", 0)
                        setup_flag = int(bool(setup_raw))
                        msg += f", setup={setup_flag}"
                    except Exception:
                        pass
                log_callback(msg)
            except Exception:
                pass
        return int(setup_pass)

    setup_pass = sum(1 for row in rows_list if bool(row.get("setup")))
    return int(setup_pass)


REQUIRED_CANDIDATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "system1": ("ROC200", "SMA25", "SMA50", "filter", "setup"),
    "system2": ("ADX7", "RSI3", "TwoDayUp", "setup"),
    "system3": ("Drop3D", "ATR10", "setup"),
    "system4": ("RSI4", "ATR40", "setup"),
    "system5": ("ADX7", "ATR10", "RSI3", "setup"),
    "system6": ("Return6D", "ATR10", "UpTwoDays", "setup"),
    "system7": ("ATR50", "setup"),
}

RANKING_CONFIG: dict[str, tuple[str | None, bool]] = {
    "system1": ("ROC200", False),
    "system2": ("ADX7", False),
    "system3": ("Drop3D", False),
    "system4": ("RSI4", True),
    "system5": ("ADX7", False),
    "system6": ("Return6D", False),
    "system7": (None, False),
}


def _resolve_top_n(system_name: str) -> int:
    try:
        return int(get_settings(create_dirs=False).backtest.top_n_rank)
    except Exception:
        return 10


def _is_fast_path_viable(
    system_name: str, prepared: dict[str, pd.DataFrame], prev_day: pd.Timestamp
) -> bool:
    if not prepared:
        return False
    try:
        prev_ts = pd.Timestamp(prev_day).normalize()
    except Exception:
        return False
    name = (system_name or "").lower()
    required_cols = set(REQUIRED_CANDIDATE_COLUMNS.get(name, tuple()))
    available_cols: set[str] = set()
    has_prev_setup = False
    for df in prepared.values():
        if df is None or getattr(df, "empty", True):
            continue
        try:
            available_cols.update(map(str, df.columns))
        except Exception:
            pass
        if "setup" in df.columns and prev_ts in df.index:
            has_prev_setup = True
    if required_cols and not required_cols.issubset(available_cols):
        return False
    return has_prev_setup


def _collect_candidates_for_today(
    system_name: str,
    prepared: dict[str, pd.DataFrame],
    today: pd.Timestamp,
    prev_day: pd.Timestamp,
    spy_df: pd.DataFrame | None,
) -> dict[pd.Timestamp, list[dict]]:
    name = system_name.lower()
    gate = (lambda _ts: True)
    if name == "system1":
        gate = _make_spy_gate(spy_df, "SMA100")
    elif name == "system4":
        gate = _make_spy_gate(spy_df, "SMA200")

    required_cols = REQUIRED_CANDIDATE_COLUMNS.get(name, tuple())
    rank_key, asc = RANKING_CONFIG.get(name, (None, True))
    top_n = _resolve_top_n(name)

    entry_map: dict[pd.Timestamp, list[dict]] = {}

    for sym, df in prepared.items():
        if "setup" not in df.columns:
            continue
        if prev_day not in df.index:
            continue
        try:
            row = df.loc[prev_day]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
        except Exception:
            continue
        try:
            setup_flag = bool(row.get("setup"))
        except Exception:
            setup_flag = False
        if not setup_flag:
            continue
        if not gate(prev_day):
            continue
        try:
            loc = df.index.get_loc(prev_day)
        except Exception:
            continue
        if isinstance(loc, slice):
            loc = loc.stop - 1
        elif isinstance(loc, np.ndarray):
            if len(loc) == 0:
                continue
            loc = int(loc[-1])
        else:
            loc = int(loc)
        if loc < 0:
            continue
        if loc + 1 < len(df.index):
            try:
                entry_date_raw = pd.Timestamp(df.index[loc + 1])
            except Exception:
                continue
        else:
            try:
                entry_date_raw = pd.Timestamp(get_next_nyse_trading_day(prev_day))
            except Exception:
                continue
        try:
            entry_date = entry_date_raw.normalize()
        except Exception:
            entry_date = entry_date_raw
        if entry_date != today:
            continue
        candidate = {
            "symbol": sym,
            "entry_date": entry_date,
            "Date": prev_day,
        }
        for col in required_cols:
            if col in row.index:
                candidate[col] = row[col]
        entry_map.setdefault(entry_date, []).append(candidate)

    if rank_key is not None:
        for date, items in list(entry_map.items()):
            def _sort_key(rec: dict) -> tuple[int, float, str]:
                raw = rec.get(rank_key)
                try:
                    num = float(raw)
                    if np.isnan(num):
                        raise ValueError
                except Exception:
                    return (1, 0.0, str(rec.get("symbol", "")))
                order_val = num if asc else -num
                return (0, order_val, str(rec.get("symbol", "")))

            sorted_items = sorted(items, key=_sort_key)
            total_count = len(sorted_items)
            for idx, rec in enumerate(sorted_items):
                rec["_rank"] = idx + 1
                rec["_total_candidates"] = total_count
            entry_map[date] = sorted_items[:top_n]
    else:
        for date, items in list(entry_map.items()):
            total_count = len(items)
            for idx, rec in enumerate(items):
                rec["_rank"] = idx + 1
                rec["_total_candidates"] = total_count
            entry_map[date] = items[:top_n]

    return entry_map


def get_today_signals_for_strategy(
    strategy,
    raw_data_dict: dict[str, pd.DataFrame],
    *,
    market_df: pd.DataFrame | None = None,
    today: pd.Timestamp | None = None,
    progress_callback: Callable[..., None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    stage_progress: (
        Callable[[int, int | None, int | None, int | None, int | None], None] | None
    ) = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """
    各 Strategy の prepare_data / generate_candidates を流用し、
    最新営業日の候補のみを DataFrame で返す。

    戻り値カラム:
        - symbol, system, side, signal_type,
          entry_date, entry_price, stop_price,
          score_key, score
    """
    from common.utils_spy import get_latest_nyse_trading_day

    try:
        system_name = str(strategy.SYSTEM_NAME).lower()  # type: ignore[attr-defined]
    except Exception:
        system_name = ""
    side = _infer_side(system_name)
    signal_type = "sell" if side == "short" else "buy"

    # 取引日
    if today is None:
        today = get_latest_nyse_trading_day()
    try:
        today_ts = pd.Timestamp(today)
    except Exception:
        today_ts = get_latest_nyse_trading_day()
    if getattr(today_ts, "tzinfo", None) is not None:
        try:
            today_ts = today_ts.tz_convert(None)
        except (TypeError, ValueError, AttributeError):
            try:
                today_ts = today_ts.tz_localize(None)
            except Exception:
                today_ts = pd.Timestamp(today_ts.to_pydatetime().replace(tzinfo=None))
    today = today_ts.normalize()

    # 準備
    total_symbols = len(raw_data_dict)
    if log_callback:
        try:
            log_callback(f"🧪 フィルターチェック開始：{total_symbols} 銘柄")
        except Exception:
            pass
    # 0% -> 25%
    try:
        if stage_progress:
            # 0% ステージでは対象銘柄数を第1引数に渡す（UI 側で "対象→n" 表示に使用）
            stage_progress(0, total_symbols, None, None, None)
    except Exception:
        pass
    t0 = _t.time()
    # ルックバック最適化：必要日数が指定されていれば各DFを末尾N行にスライス
    sliced_dict = raw_data_dict
    try:
        if (
            lookback_days is not None
            and lookback_days > 0
            and isinstance(raw_data_dict, dict)
        ):
            sliced: dict[str, pd.DataFrame] = {}
            for _sym, _df in raw_data_dict.items():
                try:
                    if _df is None or getattr(_df, "empty", True):
                        continue
                    x = _df.copy()
                    if "Date" in x.columns:
                        idx = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                        x.index = pd.Index(idx)
                    else:
                        x.index = pd.to_datetime(x.index, errors="coerce").normalize()
                    # 不正な日時は除外
                    x = x[~x.index.isna()]
                    # 末尾N営業日相当を抽出
                    x = x.tail(int(lookback_days))
                    sliced[_sym] = x
                except Exception:
                    sliced[_sym] = _df
            sliced_dict = sliced
    except Exception:
        sliced_dict = raw_data_dict

    prepared: dict[str, pd.DataFrame] = {}
    for sym, df in (sliced_dict or {}).items():
        norm = _normalize_symbol_frame(df)
        if norm is not None:
            prepared[sym] = norm

    try:
        prev_trading_day = get_latest_nyse_trading_day(
            pd.Timestamp(today) - pd.Timedelta(days=1)
        )
    except Exception:
        prev_trading_day = pd.Timestamp(today) - pd.Timedelta(days=1)

    try:
        prev_trading_day = pd.Timestamp(prev_trading_day).normalize()
    except Exception:
        prev_trading_day = pd.Timestamp(today) - pd.Timedelta(days=1)

    if not prepared or not _is_fast_path_viable(
        system_name, prepared, prev_trading_day
    ):
        try:
            if stage_progress:
                stage_progress(25, 0, None, None, None)
                stage_progress(50, 0, 0, None, None)
                stage_progress(75, 0, 0, 0, None)
        except Exception:
            pass
        if log_callback:
            try:
                log_callback(
                    "⚠️ 必要な指標が不足しているため候補抽出をスキップしました"
                )
            except Exception:
                pass
        return _empty_signals_frame()

    try:
        if log_callback:
            em, es = divmod(int(max(0, _t.time() - t0)), 60)
            log_callback(f"⏱️ フィルター/前処理 完了（経過 {em}分{es}秒）")
    except Exception:
        pass

    latest_rows, filter_pass = _build_latest_rows(prepared, prev_trading_day)
    if str(system_name).lower() == "system7":
        filter_pass = 1 if "SPY" in prepared else 0

    if log_callback:
        try:
            log_callback(f"🧪 フィルターチェック完了：{filter_pass} 銘柄")
        except Exception:
            pass
    try:
        if stage_progress:
            stage_progress(25, filter_pass, None, None, None)
    except Exception:
        pass

    spy_df_norm: pd.DataFrame | None = None
    try:
        sys_lower = str(system_name).lower()
        if sys_lower in {"system1", "system4", "system7"}:
            spy_source = (
                market_df
                if isinstance(market_df, pd.DataFrame)
                else prepared.get("SPY")
            )
            base_spy = _normalize_symbol_frame(spy_source)
            if sys_lower == "system1":
                required_col: str | None = "SMA100"
            elif sys_lower == "system4":
                required_col = "SMA200"
            else:
                required_col = None
            if base_spy is not None and (
                required_col is None or required_col in base_spy.columns
            ):
                spy_df_norm = base_spy
            elif spy_source is not None:
                spy_full = get_spy_with_indicators(spy_source)
                spy_df_norm = _normalize_symbol_frame(spy_full)
            else:
                spy_df_norm = base_spy
    except Exception:
        spy_df_norm = _normalize_symbol_frame(
            market_df if isinstance(market_df, pd.DataFrame) else prepared.get("SPY")
        )

    setup_pass = _compute_setup_pass(
        system_name,
        latest_rows,
        filter_pass,
        prev_trading_day,
        log_callback,
        spy_df_norm,
    )
    try:
        if stage_progress:
            stage_progress(50, filter_pass, setup_pass, None, None)
    except Exception:
        pass

    if log_callback:
        try:
            log_callback(f"🧩 セットアップチェック開始：{filter_pass} 銘柄")
        except Exception:
            pass

    t1 = _t.time()
    candidates_by_date = _collect_candidates_for_today(
        system_name, prepared, today, prev_trading_day, spy_df_norm
    )
    try:
        if log_callback:
            em, es = divmod(int(max(0, _t.time() - t1)), 60)
            log_callback(f"⏱️ セットアップ/候補抽出 完了（経過 {em}分{es}秒）")
    except Exception:
        pass
    # トレード候補件数（当日のみ）→ UI表示は最大ポジション数に合わせて上限10に丸める
    # 候補キー型のゆらぎ（str/date/Timestamp）を吸収するため、
    # 正規化Timestamp→元キーのマップを作成してから選択・参照する
    try:
        key_map: dict[pd.Timestamp, object] = {}
        cand_keys = list((candidates_by_date or {}).keys())
        for _k in cand_keys:
            try:
                _raw_ts = pd.to_datetime(_k, errors="coerce")
                if pd.isna(_raw_ts):
                    continue
                _ts = pd.Timestamp(_raw_ts)
                if getattr(_ts, "tzinfo", None) is not None:
                    try:
                        _ts = _ts.tz_localize(None)
                    except Exception:
                        try:
                            _ts = pd.Timestamp(
                                _ts.to_pydatetime().replace(tzinfo=None)
                            )
                        except Exception:
                            continue
                _ts = _ts.normalize()
                if _ts not in key_map:
                    key_map[_ts] = _k
            except Exception:
                continue
        candidate_dates = sorted(list(key_map.keys()), reverse=True)
    except Exception:
        key_map = {}
        candidate_dates = []

    target_date: pd.Timestamp | None = None
    fallback_reason: str | None = None

    def _collect_recent_days(
        anchor: pd.Timestamp | None, count: int
    ) -> list[pd.Timestamp]:
        if anchor is None or count <= 0:
            return []
        out: list[pd.Timestamp] = []
        seen: set[pd.Timestamp] = set()
        cur = pd.Timestamp(anchor).normalize()
        while len(out) < count:
            if cur in seen:
                break
            out.append(cur)
            seen.add(cur)
            prev = get_latest_nyse_trading_day(cur - pd.Timedelta(days=1))
            prev = pd.Timestamp(prev).normalize()
            if prev >= cur:
                break
            cur = prev
        return out

    try:
        primary_days = _collect_recent_days(today, 3)
        for dt in primary_days:
            if dt in candidate_dates:
                target_date = dt
                break

        if target_date is None:
            try:
                settings = get_settings(create_dirs=False)
                cfg = getattr(settings, "cache", None)
                rolling_cfg = getattr(cfg, "rolling", None)
                max_stale = getattr(
                    rolling_cfg,
                    "max_staleness_days",
                    getattr(rolling_cfg, "max_stale_days", 2),
                )
                stale_limit = int(max_stale)
            except Exception:
                stale_limit = 2
            fallback_window = max(len(primary_days), stale_limit + 3)
            extended_days = _collect_recent_days(today, fallback_window)
            for dt in extended_days:
                if dt in candidate_dates:
                    target_date = dt
                    if dt not in primary_days:
                        fallback_reason = "recent"
                    break

        if target_date is None and candidate_dates:
            today_norm = (
                pd.Timestamp(today).normalize() if today is not None else None
            )
            past_candidates = [
                d
                for d in candidate_dates
                if today_norm is None or d <= today_norm
            ]
            if past_candidates:
                target_date = max(past_candidates)
                if fallback_reason is None:
                    fallback_reason = "latest_past"
            else:
                target_date = max(candidate_dates)
                if fallback_reason is None:
                    fallback_reason = "latest_any"

        if log_callback:
            try:
                _cands_str = ", ".join([str(d.date()) for d in candidate_dates[:5]])
                _search_str = ", ".join([str(d.date()) for d in primary_days])
                _chosen = str(target_date.date()) if target_date is not None else "None"
                fallback_msg = ""
                if fallback_reason:
                    fallback_labels = {
                        "recent": "直近営業日に候補が無いため過去日を採用",
                        "latest_past": "探索範囲外の最新過去日を採用",
                        "latest_any": "未来日しか存在しないため候補最終日を採用",
                    }
                    label = fallback_labels.get(fallback_reason, fallback_reason)
                    fallback_msg = f" | フォールバック: {label}"
                log_callback(
                    "🗓️ 候補日（最新上位）: "
                    f"{_cands_str} | 探索順: {_search_str} | 採用: {_chosen}{fallback_msg}"
                )
            except Exception:
                pass
    except Exception:
        target_date = None
        fallback_reason = None
    try:
        if target_date is not None and target_date in key_map:
            orig_key = key_map[target_date]
            total_candidates_today = len(
                (candidates_by_date or {}).get(orig_key, []) or []
            )
        else:
            total_candidates_today = 0
    except Exception:
        total_candidates_today = 0
    # UIのTRDlistは各systemの最大ポジション数を超えないように表示
    try:
        _max_pos_ui = int(get_settings(create_dirs=False).risk.max_positions)
    except Exception:
        _max_pos_ui = 10
    if total_candidates_today and _max_pos_ui > 0:
        total_candidates_today = min(int(total_candidates_today), int(_max_pos_ui))
    try:
        if stage_progress:
            stage_progress(75, filter_pass, setup_pass, total_candidates_today, None)
    except Exception:
        pass
    if log_callback:
        try:
            log_callback(f"🧩 セットアップチェック完了：{setup_pass} 銘柄")
            # 誤解回避: ここでの件数は『候補生成の母集団（セットアップ通過）』
            log_callback(f"🧮 候補生成元（セットアップ通過）：{setup_pass} 銘柄")
            # TRDlist 相当（直近営業日時点の候補数。最大{_max_pos_ui}に丸め）
            log_callback(
                f"🧮 TRDlist相当（直近営業日時点の候補数）：{total_candidates_today} 銘柄"
            )
        except Exception:
            pass

    if not candidates_by_date:
        return _empty_signals_frame()

    # 当日または直近過去日の候補のみ抽出
    if target_date is not None and target_date in key_map:
        orig_key2 = key_map[target_date]
        today_candidates = cast(
            list[dict], candidates_by_date.get(orig_key2, [])
        )
    else:
        today_candidates = cast(list[dict], [])
    if not today_candidates:
        return _empty_signals_frame()
    rows: list[TodaySignal] = []
    for c in today_candidates:
        sym = c.get("symbol")
        if not sym or sym not in prepared:
            continue
        df = prepared[sym]
        comp = _compute_entry_stop(strategy, df, c, side)
        if not comp:
            continue
        entry, stop = comp
        skey, sval, _asc = _score_from_candidate(system_name, c)

        # System1 は ROC200 を必ずスコアに採用できるよう堅牢化
        try:
            if (system_name == "system1") and (
                skey is None or str(skey).upper() != "ROC200"
            ):
                skey = "ROC200"
        except Exception:
            pass

        if skey is not None and (
            sval is None or (isinstance(sval, float) and pd.isna(sval))
        ):
            try:
                raw_val = c.get(skey)
                if raw_val is not None:
                    num = float(raw_val)
                    if not np.isnan(num):
                        sval = num
            except Exception:
                pass

        rank_val: int | None = None
        try:
            rank_raw = c.get("_rank")
            if rank_raw is not None and not pd.isna(rank_raw):
                rank_val = int(rank_raw)
        except Exception:
            rank_val = None

        try:
            total_for_rank = int(c.get("_total_candidates", 0))
        except Exception:
            total_for_rank = 0
        if total_for_rank <= 0:
            total_for_rank = len(today_candidates)

        if (
            sval is None or (isinstance(sval, float) and pd.isna(sval))
        ) and rank_val is not None:
            try:
                sval = float(rank_val)
            except Exception:
                pass

        # 選定理由（順位を最優先、なければ簡潔かつシステム固有の文言）
        reason_parts: list[str] = []
        # System1 は日本語で「ROC200がn位のため」に統一（順位が取れない場合のみ汎用文言）
        if system_name == "system1":
            if rank_val is not None and int(rank_val) <= 10:
                reason_parts = [f"ROC200が{int(rank_val)}位のため"]
            else:
                reason_parts = ["ROC200が上位のため"]
        elif system_name == "system2":
            if rank_val is not None and skey is not None:
                reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
            else:
                reason_parts = ["モメンタムが強く過熱のため"]
        elif system_name == "system3":
            if rank_val is not None and skey is not None:
                reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
            else:
                reason_parts = ["ボラティリティが高く条件一致のため"]
        elif system_name == "system4":
            if rank_val is not None:
                reason_parts = [f"RSI4が{rank_val}位（低水準）のため"]
            else:
                reason_parts = ["SPY上昇局面の押し目候補のため"]
        elif system_name == "system5":
            if rank_val is not None and skey is not None:
                reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
            else:
                reason_parts = ["ADXが強く、反発期待のため"]
        elif system_name == "system6":
            if rank_val is not None:
                reason_parts = [f"過去6日騰落率が{rank_val}位のため"]
            else:
                reason_parts = ["短期下落トレンド（ショート）条件一致のため"]
        elif system_name == "system7":
            # ATR50 は損切り計算用。理由は「50日安値更新」に限定する。
            reason_parts = ["SPYが50日安値を更新したため（ヘッジ）"]
        else:
            if skey is not None and rank_val is not None:
                if rank_val <= 10:
                    reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
                else:
                    reason_parts = [f"rank={rank_val}/{total_for_rank}"]
            elif skey is not None:
                # 値は原則非表示（冗長回避）。必要最小限だけ示す。
                try:
                    if sval is not None and not (
                        isinstance(sval, float) and pd.isna(sval)
                    ):
                        reason_parts.append("スコア条件を満たしたため")
                except Exception:
                    reason_parts.append("スコア条件を満たしたため")

        # fallback generic info
        if not reason_parts:
            reason_parts.append("条件一致のため")

        reason_text = "; ".join(reason_parts)

        try:
            _ed_raw: Any = c.get("entry_date")
            _ed = pd.Timestamp(_ed_raw) if _ed_raw is not None else None
            if _ed is None or pd.isna(_ed):
                # entry_date が欠損する候補は無効
                continue
            entry_date_norm = pd.Timestamp(_ed).normalize()
        except Exception:
            continue

        rows.append(
            TodaySignal(
                symbol=str(sym),
                system=system_name,
                side=side,
                signal_type=signal_type,
                entry_date=entry_date_norm,
                entry_price=float(entry),
                stop_price=float(stop),
                score_key=skey,
                # スコアは値があれば値、無ければ順位（上記で補完済み）
                score=(
                    None
                    if sval is None or (isinstance(sval, float) and pd.isna(sval))
                    else float(sval)
                ),
                reason=reason_text,
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "system",
                "side",
                "signal_type",
                "entry_date",
                "entry_price",
                "stop_price",
                "score_key",
                "score",
            ]
        )

    out = pd.DataFrame([r.__dict__ for r in rows])

    try:
        max_pos = int(get_settings(create_dirs=False).risk.max_positions)
    except Exception:
        max_pos = 10
    if max_pos > 0 and not out.empty:

        def _sort_val(row: pd.Series) -> float:
            sc = row.get("score")
            sk = row.get("score_key")
            if sc is None or (isinstance(sc, float) and pd.isna(sc)):
                return float("inf")
            return float(sc) if _asc_by_score_key(sk) else -float(sc)

        out["_sort_val"] = out.apply(_sort_val, axis=1)
        out = (
            out.sort_values("_sort_val")
            .head(max_pos)
            .drop(columns=["_sort_val"])
            .reset_index(drop=True)
        )
    final_count = len(out)

    try:
        if log_callback:
            log_callback(f"🧮 トレード候補選定完了（当日）：{final_count} 銘柄")
    except Exception:
        pass
    try:
        if stage_progress:
            stage_progress(
                100, filter_pass, setup_pass, total_candidates_today, final_count
            )
    except Exception:
        pass
    return out


def run_all_systems_today(
    symbols: list[str] | None,
    *,
    slots_long: int | None = None,
    slots_short: int | None = None,
    capital_long: float | None = None,
    capital_short: float | None = None,
    save_csv: bool = False,
    csv_name_mode: str | None = None,
    notify: bool = True,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    per_system_progress: Callable[[str, str], None] | None = None,
    symbol_data: dict[str, pd.DataFrame] | None = None,
    parallel: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """scripts.run_all_systems_today.compute_today_signals のラッパー。"""
    from scripts.run_all_systems_today import compute_today_signals as _compute

    return _compute(
        symbols,
        slots_long=slots_long,
        slots_short=slots_short,
        capital_long=capital_long,
        capital_short=capital_short,
        save_csv=save_csv,
        csv_name_mode=csv_name_mode,
        notify=notify,
        log_callback=log_callback,
        progress_callback=progress_callback,
        per_system_progress=per_system_progress,
        symbol_data=symbol_data,
        parallel=parallel,
    )


compute_today_signals = run_all_systems_today


__all__ = [
    "get_today_signals_for_strategy",
    "LONG_SYSTEMS",
    "SHORT_SYSTEMS",
    "run_all_systems_today",
    "compute_today_signals",
]
