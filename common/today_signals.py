from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd


# --- サイド定義（売買区分）---
LONG_SYSTEMS = {"system1", "system3", "system4", "system5"}
SHORT_SYSTEMS = {"system2", "system6", "system7"}


@dataclass(frozen=True)
class TodaySignal:
    symbol: str
    system: str
    side: str  # "long" | "short"
    signal_type: str  # "buy" | "sell"
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    score_key: Optional[str] = None
    score: Optional[float] = None


def _infer_side(system_name: str) -> str:
    name = (system_name or "").lower()
    if name in SHORT_SYSTEMS:
        return "short"
    return "long"


def _score_from_candidate(system_name: str, candidate: dict) -> Tuple[Optional[str], Optional[float], bool]:
    """
    候補レコードからスコア項目と並び順（昇順か）を推定して返す。
    戻り値: (score_key, score_value, asc)
    """
    name = (system_name or "").lower()
    # システム別の代表スコア
    key_order: List[Tuple[List[str], bool]] = [
        (["ROC200"], False),  # s1: 大きいほど良い
        (["ADX7"], False),    # s2,s5: 大きいほど良い
        (["DropRate_3D"], False),  # s3: 大きいほど良い（下落率）
        (["RSI4"], True),     # s4: 小さいほど良い
        (["Return6D"], False),# s6: 大きいほど良い
        (["ATR50"], False),   # s7: 参考
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
                try:
                    return k, float(candidate.get(k)), asc
                except Exception:
                    return k, None, asc
    # 見つからない場合
    return None, None, False


def _pick_atr_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("ATR20", "ATR10", "ATR40", "ATR50", "ATR14"):
        if col in df.columns:
            return col
    return None


def _compute_entry_stop(strategy, df: pd.DataFrame, candidate: dict, side: str) -> Optional[Tuple[float, float]]:
    # strategy 独自の compute_entry があれば優先
    if hasattr(strategy, "compute_entry") and callable(getattr(strategy, "compute_entry")):
        try:
            res = strategy.compute_entry(df, candidate, 0.0)
            if res and isinstance(res, tuple) and len(res) == 2:
                entry, stop = float(res[0]), float(res[1])
                if entry > 0 and (side == "short" and stop > entry or side == "long" and entry > stop):
                    return round(entry, 4), round(stop, 4)
        except Exception:
            pass

    # フォールバック: 当日始値 ± 3*ATR
    try:
        entry_idx = df.index.get_loc(candidate["entry_date"])  # type: ignore[index]
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


def get_today_signals_for_strategy(
    strategy,
    raw_data_dict: Dict[str, pd.DataFrame],
    *,
    market_df: Optional[pd.DataFrame] = None,
    today: Optional[pd.Timestamp] = None,
    progress_callback: Optional[Callable[..., None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    各 Strategy の prepare_data / generate_candidates を流用し、
    最新営業日の候補のみを DataFrame で返す。

    戻り値カラム:
      - symbol, system, side, signal_type, entry_date, entry_price, stop_price, score_key, score
    """
    from common.utils_spy import get_latest_nyse_trading_day

    system_name = getattr(strategy, "SYSTEM_NAME", "").lower()
    side = _infer_side(system_name)
    signal_type = "sell" if side == "short" else "buy"

    # 取引日
    if today is None:
        today = get_latest_nyse_trading_day()
    if isinstance(today, pd.Timestamp):
        today = today.normalize()

    # 準備
    prepared = strategy.prepare_data(
        raw_data_dict,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )

    # 候補生成（market_df を必要とする実装に配慮）
    gen_fn = getattr(strategy, "generate_candidates")
    params = inspect.signature(gen_fn).parameters
    if "market_df" in params and market_df is not None:
        candidates_by_date, _ = gen_fn(
            prepared,
            market_df=market_df,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
    else:
        candidates_by_date, _ = gen_fn(
            prepared,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

    if not candidates_by_date:
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

    # 当日分のみ抽出
    today_candidates: List[dict] = candidates_by_date.get(today, [])  # type: ignore[index]
    if not today_candidates:
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

    rows: List[TodaySignal] = []
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
        rows.append(
            TodaySignal(
                symbol=str(sym),
                system=system_name,
                side=side,
                signal_type=signal_type,
                entry_date=pd.Timestamp(c.get("entry_date")).normalize(),  # type: ignore[arg-type]
                entry_price=float(entry),
                stop_price=float(stop),
                score_key=skey,
                score=sval,
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
    return out


__all__ = [
    "get_today_signals_for_strategy",
    "LONG_SYSTEMS",
    "SHORT_SYSTEMS",
]

