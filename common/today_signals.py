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


def _compute_entry_stop(
    strategy, df: pd.DataFrame, candidate: dict, side: str
) -> Optional[Tuple[float, float]]:
    # strategy 独自の compute_entry があれば優先
    if hasattr(strategy, "compute_entry") and callable(getattr(strategy, "compute_entry")):
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
            int(bool(getattr(df, "empty", True) is False and "filter" in df.columns and bool(pd.Series(df["filter"]).tail(1).iloc[0])))
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
        total_candidates = sum(len(v or []) for v in (candidates_by_date or {}).values())
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
        # build human-readable reason（score のフォールバック補完を実施）
        reason_parts: List[str] = []
        # 欠損・NaN の場合は prepared 側から採取（同一シグナル日の値）
        if skey is not None and (sval is None or (isinstance(sval, float) and pd.isna(sval))):
            try:
                # signal day は entry_date の前営業日
                signal_date = pd.Timestamp(c.get("Date", None))
                if signal_date is None or pd.isna(signal_date):
                    signal_date = pd.Timestamp(c.get("entry_date")).normalize() - pd.Timedelta(days=1)
                if "Date" in df.columns:
                    row = df[pd.to_datetime(df["Date"]).dt.normalize() == pd.to_datetime(signal_date).normalize()]
                else:
                    row = df[pd.to_datetime(df.index).normalize() == pd.to_datetime(signal_date).normalize()]
                if not row.empty and skey in row.columns:
                    _v = row.iloc[0][skey]
                    if _v is not None and not pd.isna(_v):
                        sval = float(_v)
            except Exception:
                pass

        if skey is not None:
            # 一旦数値を整形
            try:
                sval_disp = f"{float(sval):.2f}" if sval is not None and not pd.isna(sval) else "nan"
            except Exception:
                sval_disp = str(sval)
            reason_parts.append(f"{skey}={sval_disp}")
            # 同一エントリー日における順位（上位10位なら自然文表記に）
            try:
                entry_date_norm = pd.Timestamp(c.get("entry_date")).normalize()
                vals: List[float] = []
                for psym, pdf in prepared.items():
                    try:
                        if "Date" in pdf.columns:
                            row = pdf[pd.to_datetime(pdf["Date"]).dt.normalize() == entry_date_norm]
                        else:
                            row = pdf[pd.to_datetime(pdf.index).normalize() == entry_date_norm]
                        if not row.empty and skey in row.columns:
                            v = row.iloc[0][skey]
                            if v is not None and not pd.isna(v):
                                vals.append(float(v))
                    except Exception:
                        continue
                rank = None
                total = len(vals)
                if total:
                    # 値が未設定なら候補値を使う
                    try:
                        candidate_val = float(sval) if sval is not None else None
                    except Exception:
                        candidate_val = None
                    if candidate_val is not None:
                        sorted_vals = sorted(vals, reverse=not _asc)
                        try:
                            rank = sorted_vals.index(candidate_val) + 1
                        except ValueError:
                            diffs = [abs(candidate_val - x) for x in sorted_vals]
                            rank = diffs.index(min(diffs)) + 1
                # rank に応じて自然文へ
                if rank is not None and rank <= 10:
                    reason_parts = [f"{skey}が{rank}位のため"]
                elif rank is not None and total:
                    reason_parts.append(f"rank={rank}/{total}")
            except Exception:
                pass

        # fallback generic info
        if not reason_parts:
            # include keys present in candidate for debugging
            try:
                keys = ", ".join(
                    f"{k}:{v}" for k, v in c.items() if k not in {"symbol", "entry_date"}
                )
                reason_parts.append(keys[:500])
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
                score=sval,
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
