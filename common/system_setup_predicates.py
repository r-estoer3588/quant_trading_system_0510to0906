"""Shared setup predicate helpers for Systems 1,3,4,5 (Phase2 refactor).

目的:
    - 各 System の最終 *setup* 条件を 1 か所に集約し、後段 (candidate 生成 / diagnostics) で
      一貫したロジックを利用できるようにする。
    - 既存 core/systemX.py 内の `filter` / `setup` 列生成ロジックと**同値**になるよう注意。
    - NaN / 欠損 / 型異常時はただちに False を返し、副作用を発生させない (純関数)。

まだ利用箇所は差し替えていない (ID7 で統合予定)。

設計指針:
    - 個別関数: system1_setup_predicate(row: pd.Series) -> bool など
    - 追加便利関数: get_system_setup_predicate(name_or_id) -> Callable
    - System5 の ATR% 閾値は core/system5.py の DEFAULT_ATR_PCT_THRESHOLD と同値 (循環依存を避けるため値をローカル再定義)。
    - System 名 / ID いずれでも取得できる軽量マップ SETUP_PREDICATES を提供。

後続 (ID8 diagnostics enrichment) で以下カウンタに利用予定:
    - setup_predicate_count
    - final_top_n_count
    - mismatch_flag

注意:
    - System6 (既に同期済) / System2(ショート) / System7 は別 ID で対応予定。
    - 既存コードに影響を与えないため外部 import を控え、標準構造 + pandas のみ依存。
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, Any
import math
import pandas as pd
import os
import random

# 型ヒント用
from typing import Optional

# System5 の ATR% 閾値: core/system5.DEFAULT_ATR_PCT_THRESHOLD と揃える (循環依存回避のため再定義)
DEFAULT_ATR_PCT_THRESHOLD: float = 0.025


# --- 汎用ユーティリティ ----------------------------------------------------


def _to_float(value: Any) -> float:
    """安全な float 変換 (失敗 / NaN は math.nan)。"""
    try:
        v = float(value)
        if math.isnan(v):
            return math.nan
        return v
    except Exception:
        return math.nan


def _all_not_nan(values: list[float]) -> bool:
    return all((not math.isnan(v)) for v in values)


# --- System1 -----------------------------------------------------------------
# 条件: Close>=5, dollarvolume20>25M, Close>sma200, roc200>0
# (filter + setup を 1 つに合成: 元コードでの filter & setup 連鎖と同値)


def system1_setup_predicate(row: pd.Series) -> bool:
    try:
        close = _to_float(row.get("Close"))
        dv20 = _to_float(row.get("dollarvolume20"))
        sma200 = _to_float(row.get("sma200"))
        roc200 = _to_float(row.get("roc200"))
        if not _all_not_nan([close, dv20, sma200, roc200]):
            return False
        return (close >= 5.0) and (dv20 > 25_000_000) and (close > sma200) and (roc200 > 0.0)
    except Exception:
        return False


# --- System3 -----------------------------------------------------------------
# 条件: Close>=5, dollarvolume20>25M, atr_ratio>=0.05, drop3d>=0.125


def system3_setup_predicate(row: pd.Series) -> bool:
    try:
        close = _to_float(row.get("Close"))
        dv20 = _to_float(row.get("dollarvolume20"))
        atr_ratio = _to_float(row.get("atr_ratio"))
        drop3d = _to_float(row.get("drop3d"))
        if not _all_not_nan([close, dv20, atr_ratio, drop3d]):
            return False
        return (close >= 5.0) and (dv20 > 25_000_000) and (atr_ratio >= 0.05) and (drop3d >= 0.125)
    except Exception:
        return False


# --- System2 (Short spike) ---------------------------------------------------
# 条件: Close>=5, dollarvolume20>25M, atr_ratio>0.03, rsi3>90, twodayup==True
def system2_setup_predicate(row: pd.Series) -> bool:
    try:
        close = _to_float(row.get("Close"))
        dv20 = _to_float(row.get("dollarvolume20"))
        atr_ratio = _to_float(row.get("atr_ratio"))
        rsi3 = _to_float(row.get("rsi3"))
        two_up = bool(row.get("twodayup"))
        if not _all_not_nan([close, dv20, atr_ratio, rsi3]):
            return False
        return (
            (close >= 5.0)
            and (dv20 > 25_000_000)
            and (atr_ratio > 0.03)
            and (rsi3 > 90.0)
            and two_up
        )
    except Exception:
        return False


# --- System4 -----------------------------------------------------------------
# 条件: dollarvolume50>100M, hv50 10-40%, Close>sma200


def system4_setup_predicate(row: pd.Series) -> bool:
    try:
        dv50 = _to_float(row.get("dollarvolume50"))
        hv50 = _to_float(row.get("hv50"))
        close = _to_float(row.get("Close"))
        sma200 = _to_float(row.get("sma200"))
        if not _all_not_nan([dv50, hv50, close, sma200]):
            return False
        return (dv50 > 100_000_000) and (10.0 <= hv50 <= 40.0) and (close > sma200)
    except Exception:
        return False


# --- System5 -----------------------------------------------------------------
# 条件 (filter == setup): Close>=5, adx7>35, atr_pct>DEFAULT_ATR_PCT_THRESHOLD


def system5_setup_predicate(row: pd.Series, *, atr_pct_threshold: float | None = None) -> bool:
    try:
        close = _to_float(row.get("Close"))
        adx7 = _to_float(row.get("adx7"))
        atr_pct = _to_float(row.get("atr_pct"))
        threshold = (
            atr_pct_threshold if atr_pct_threshold is not None else DEFAULT_ATR_PCT_THRESHOLD
        )
        if not _all_not_nan([close, adx7, atr_pct]):
            return False
        return (close >= 5.0) and (adx7 > 35.0) and (atr_pct > threshold)
    except Exception:
        return False


# --- System7 (SPY hedge) -----------------------------------------------------
# 条件: Low <= min_50  (core/system7.py と同値。最終候補は翌営業日エントリー用に別日扱い)
def system7_setup_predicate(row: pd.Series) -> bool:
    try:
        low_v = _to_float(row.get("Low"))
        min50 = _to_float(row.get("min_50"))
        if math.isnan(low_v) or math.isnan(min50):
            return False
        return low_v <= min50
    except Exception:
        return False


# --- 取得ヘルパ --------------------------------------------------------------
SETUP_PREDICATES: Mapping[str, Callable[..., bool]] = {
    # 名前 / ID どちらでも取り出せるよう重複キーを用意
    "1": system1_setup_predicate,
    "System1": system1_setup_predicate,
    "2": system2_setup_predicate,
    "System2": system2_setup_predicate,
    "3": system3_setup_predicate,
    "System3": system3_setup_predicate,
    "4": system4_setup_predicate,
    "System4": system4_setup_predicate,
    "5": system5_setup_predicate,
    "System5": system5_setup_predicate,
    "7": system7_setup_predicate,
    "System7": system7_setup_predicate,
}


def get_system_setup_predicate(name_or_id: str) -> Callable[..., bool]:
    """Return setup predicate function for a system.

    未定義 ID の場合はラムダ False を返す (呼び出し側で個別エラーにしない方針)。
    後続統合ステップで未知システム利用は diagnostics にも記録予定。
    """
    return SETUP_PREDICATES.get(name_or_id, lambda *_a, **_k: False)


__all__ = [
    "system1_setup_predicate",
    "system2_setup_predicate",
    "system3_setup_predicate",
    "system4_setup_predicate",
    "system5_setup_predicate",
    "system7_setup_predicate",
    "get_system_setup_predicate",
    "SETUP_PREDICATES",
    "DEFAULT_ATR_PCT_THRESHOLD",
]


# --- 検証ユーティリティ ------------------------------------------------------
def validate_predicate_equivalence(
    prepared_dict: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
    system_id: str,
    *,
    sample_max: int = 200,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """`setup` 列と新 predicate の一致をサンプリング検証。

    環境変数 VALIDATE_SETUP_PREDICATE が "1" 系以外なら何もしない。
    速度影響を避けるため最大 sample_max 行までランダム抽出。

    ログ形式 (不一致時):
        [SystemX] setup predicate mismatch: mismatches=N sample=[SYM1,SYM2,...]
    """
    if os.environ.get("VALIDATE_SETUP_PREDICATE", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return

    pred_fn = get_system_setup_predicate(system_id)

    # 統一イテレーション: prepared_dict が dict でない場合はそのまま列挙
    if isinstance(prepared_dict, Mapping):
        items = list(prepared_dict.items())
    else:
        items = [(f"_{i}", df) for i, df in enumerate(prepared_dict)]

    # 行収集 (setup 列が存在する行のみ対象)
    rows: list[tuple[str, pd.Series]] = []
    for sym, df in items:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if "setup" not in df.columns:
            continue  # まだ列未生成 (呼び出し位置前) の場合はスキップ
        # 過剰収集防止: まず最大 sample_max*2 までは連続取得、その後 break
        for _idx, row in df.tail(sample_max * 2).iterrows():  # 最新側優先 (当日想定)
            rows.append((sym, row))
            if len(rows) >= sample_max * 2:
                break
        if len(rows) >= sample_max * 2:
            break

    if not rows:
        return

    # ランダムサンプル (行数多い場合)
    if len(rows) > sample_max:
        random.shuffle(rows)
        rows = rows[:sample_max]

    mismatches: list[str] = []
    for sym, row in rows:
        try:
            col_val = bool(row.get("setup", False))
            pred_val = bool(pred_fn(row))
        except Exception:
            continue
        if col_val != pred_val:
            mismatches.append(sym)
            if len(mismatches) >= 8:  # ログ量抑制
                break

    if mismatches and log_fn:
        try:
            log_fn(
                f"[{system_id}] setup predicate mismatch: mismatches={len(mismatches)} sample={mismatches}"
            )
        except Exception:
            pass


__all__.append("validate_predicate_equivalence")
