from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

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
    reason: Optional[str] = None


def _infer_side(system_name: str) -> str:
    name = (system_name or "").lower()
    if name in SHORT_SYSTEMS:
        return "short"
    return "long"


def _score_from_candidate(
    system_name: str, candidate: dict
) -> Tuple[Optional[str], Optional[float], bool]:
    """
    候補レコードからスコア項目と並び順（昇順か）を推定して返す。
    戻り値: (score_key, score_value, asc)
    """
    name = (system_name or "").lower()
    # システム別の代表スコア
    key_order: List[Tuple[List[str], bool]] = [
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


def _pick_atr_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("ATR20", "ATR10", "ATR40", "ATR50", "ATR14"):
        if col in df.columns:
            return col
    return None


def _compute_entry_stop(
    strategy, df: pd.DataFrame, candidate: dict, side: str
) -> Optional[Tuple[float, float]]:
    # strategy 独自の compute_entry があれば優先
    if hasattr(strategy, "compute_entry") and callable(
        getattr(strategy, "compute_entry")
    ):
        try:
            res = strategy.compute_entry(df, candidate, 0.0)
            if res and isinstance(res, tuple) and len(res) == 2:
                entry, stop = float(res[0]), float(res[1])
                if entry > 0 and (
                    side == "short" and stop > entry or side == "long" and entry > stop
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
        - symbol, system, side, signal_type,
          entry_date, entry_price, stop_price,
          score_key, score
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
    if log_callback:
        try:
            log_callback("🧪 フィルターチェック開始")
        except Exception:
            pass
    prepared = strategy.prepare_data(
        raw_data_dict,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )
    # 簡易フィルター通過件数（列がある場合のみカウント）
    try:
        filter_pass = sum(
            int(
                bool(
                    getattr(df, "empty", True) is False
                    and "filter" in df.columns
                    and bool(pd.Series(df["filter"]).tail(1).iloc[0])
                )
            )
            for df in prepared.values()
        )
    except Exception:
        filter_pass = 0
    if log_callback:
        try:
            log_callback(f"✅ フィルター通過銘柄: {filter_pass} 件")
        except Exception:
            pass

    # 候補生成（market_df を必要とする実装に配慮）
    gen_fn = getattr(strategy, "generate_candidates")
    params = inspect.signature(gen_fn).parameters
    if log_callback:
        try:
            log_callback("🧩 セットアップチェック開始")
        except Exception:
            pass
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

    # セットアップ通過件数（列がある場合のみカウント）
    try:
        setup_pass = sum(
            1
            for df in prepared.values()
            if getattr(df, "empty", True) is False
            and "setup" in df.columns
            and bool(pd.Series(df["setup"]).tail(1).iloc[0])
        )
    except Exception:
        setup_pass = 0
    # トレード候補全体件数
    try:
        total_candidates = sum(
            len(v or []) for v in (candidates_by_date or {}).values()
        )
    except Exception:
        total_candidates = 0
    if log_callback:
        try:
            log_callback(f"✅ セットアップクリア銘柄: {setup_pass} 件")
            log_callback("🧮 トレード候補選定完了")
        except Exception:
            pass

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

        # signal 日（通常は entry_date の前営業日を想定）
        signal_date_ts: Optional[pd.Timestamp] = None
        try:
            # candidate["Date"] があれば優先
            if "Date" in c and c.get("Date") is not None:
                date_arg: Any = c.get("Date")
                tmp = pd.to_datetime(date_arg, errors="coerce")
                if not pd.isna(tmp):
                    signal_date_ts = pd.Timestamp(tmp).normalize()
        except Exception:
            # フォールバックは後段の entry_date 補完に任せる
            pass
        if signal_date_ts is None:
            try:
                ed_arg: Any = c.get("entry_date")
                ed = pd.to_datetime(ed_arg, errors="coerce")
                if isinstance(ed, pd.Timestamp) and not pd.isna(ed):
                    signal_date_ts = ed.normalize() - pd.Timedelta(days=1)
            except Exception:
                signal_date_ts = None

        # 欠損スコアの補完（まず値、次に順位）
        rank_val: Optional[int] = None
        total_for_rank: int = 0
        if skey is not None:
            # 1) 欠損なら prepared から同日値を補完
            if sval is None or (isinstance(sval, float) and pd.isna(sval)):
                try:
                    if signal_date_ts is not None:
                        row = prepared[sym][
                            pd.to_datetime(
                                prepared[sym]["Date"]
                                if "Date" in prepared[sym].columns
                                else prepared[sym].index
                            )
                            .normalize()
                            .eq(signal_date_ts)
                        ]
                        if not row.empty and skey in row.columns:
                            _v = row.iloc[0][skey]
                            if _v is not None and not pd.isna(_v):
                                sval = float(_v)
                except Exception:
                    pass

            # 2) 値がまだ欠損なら、同日全銘柄の順位を算出してスコアに設定
            try:
                if signal_date_ts is not None:
                    vals: list[tuple[str, float]] = []
                    for psym, pdf in prepared.items():
                        try:
                            if "Date" in pdf.columns:
                                row = pdf[
                                    pd.to_datetime(pdf["Date"]).dt.normalize()
                                    == signal_date_ts
                                ]
                            else:
                                row = pdf[
                                    pd.to_datetime(pdf.index).normalize()
                                    == signal_date_ts
                                ]
                            if not row.empty and skey in row.columns:
                                v = row.iloc[0][skey]
                                if v is not None and not pd.isna(v):
                                    vals.append((psym, float(v)))
                        except Exception:
                            continue
                    total_for_rank = len(vals)
                    if total_for_rank:
                        # 並び順: system の昇降順推定に合わせる（ROC200 などは降順）
                        reverse = not _asc
                        # 値が同一のときはシンボルで安定ソート
                        vals_sorted = sorted(
                            vals, key=lambda t: (t[1], t[0]), reverse=reverse
                        )
                        # 自銘柄の順位を決定
                        symbols_sorted = [s for s, _ in vals_sorted]
                        if sym in symbols_sorted:
                            rank_val = symbols_sorted.index(sym) + 1
                        # スコアが欠損なら順位をそのままスコアに採用
                        if sval is None or (isinstance(sval, float) and pd.isna(sval)):
                            if rank_val is not None:
                                sval = float(rank_val)
            except Exception:
                pass

        # 選定理由（順位を最優先、なければ値）
        reason_parts: List[str] = []
        if skey is not None and rank_val is not None:
            if rank_val <= 10:
                reason_parts = [f"{skey}が{rank_val}位のため"]
            else:
                reason_parts = [f"rank={rank_val}/{total_for_rank}"]
        elif skey is not None:
            # 値があれば値を表示（nan は避ける）
            try:
                if sval is not None and not (isinstance(sval, float) and pd.isna(sval)):
                    reason_parts.append(f"{skey}={float(sval):.2f}")
            except Exception:
                if sval is not None:
                    reason_parts.append(f"{skey}={sval}")

        # fallback generic info
        if not reason_parts:
            try:
                keys = ", ".join(
                    f"{k}:{v}"
                    for k, v in c.items()
                    if k not in {"symbol", "entry_date"}
                )
                reason_parts.append(keys[:500] or "selected")
            except Exception:
                reason_parts.append("selected")

        reason_text = "; ".join(reason_parts)

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
    return out


__all__ = [
    "get_today_signals_for_strategy",
    "LONG_SYSTEMS",
    "SHORT_SYSTEMS",
]
