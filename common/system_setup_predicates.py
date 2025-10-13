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
    - ranked_top_n_count
    - mismatch_flag

注意:
    - System6 (既に同期済) / System2(ショート) / System7 は別 ID で対応予定。
    - 既存コードに影響を与えないため外部 import を控え、標準構造 + pandas のみ依存。
"""

from __future__ import annotations

import math
import os
import random

# 型ヒント用
from typing import Any, Callable, Mapping, Optional, Sequence

import pandas as pd

# Indicator access helper for case-insensitive column lookup
from common.indicator_access import get_indicator
from common.indicator_access import to_float as indicator_to_float

# System5 の ATR% 閾値: core/system5.DEFAULT_ATR_PCT_THRESHOLD と揃える (循環依存回避のため再定義)
DEFAULT_ATR_PCT_THRESHOLD: float = 0.025


# --- 汎用ユーティリティ ----------------------------------------------------


def _to_float(value: Any) -> float:
    """安全な float 変換 (失敗 / NaN は math.nan)。

    Note: This is a legacy wrapper. For indicator values, prefer using
    get_indicator() + indicator_to_float() for case-insensitive access.
    """
    return indicator_to_float(value)


def _all_not_nan(values: list[float]) -> bool:
    return all((not math.isnan(v)) for v in values)


# --- System1 -----------------------------------------------------------------
# Phase 2 filter: Close>=5, dollarvolume20>=50M
# Phase 6 setup: SMA25>SMA50, ROC200>0
# This predicate combines both for complete evaluation


def system1_setup_predicate(row: pd.Series) -> bool:
    try:
        # Phase 2 filter (redundant safety check) - use case-insensitive access
        close = indicator_to_float(get_indicator(row, "Close"))  # type: ignore[arg-type]
        dv20 = indicator_to_float(get_indicator(row, "dollarvolume20"))  # type: ignore[arg-type]
        if math.isnan(close) or math.isnan(dv20):
            return False
        if not (close >= 5.0 and dv20 >= 50_000_000):
            return False

        # Phase 6 setup (SMA trend + ROC200 momentum)
        sma25 = indicator_to_float(get_indicator(row, "sma25"))  # type: ignore[arg-type]
        sma50 = indicator_to_float(get_indicator(row, "sma50"))  # type: ignore[arg-type]
        roc200 = indicator_to_float(get_indicator(row, "roc200"))  # type: ignore[arg-type]
        if math.isnan(sma25) or math.isnan(sma50) or math.isnan(roc200):
            return False
        return (sma25 > sma50) and (roc200 > 0.0)
    except Exception:
        return False


# --- System3 -----------------------------------------------------------------
# Phase 2 filter: Low>=1, AvgVol50>=1M, atr_ratio>=0.05
# Phase 6 setup: Close>SMA150, drop3d>=0.125
# This predicate combines both for complete evaluation


def system3_setup_predicate(row: pd.Series) -> bool:
    try:
        # Phase 2 filter (safety check)
        low = indicator_to_float(get_indicator(row, "Low"))  # type: ignore[arg-type]
        avgvol50 = indicator_to_float(
            get_indicator(row, "AvgVolume50")  # type: ignore[arg-type]
        )
        atr_ratio = indicator_to_float(
            get_indicator(row, "atr_ratio")  # type: ignore[arg-type]
        )

        if math.isnan(low) or math.isnan(avgvol50) or math.isnan(atr_ratio):
            return False
        if not (low >= 1.0 and avgvol50 >= 1_000_000 and atr_ratio >= 0.05):
            return False

        # Phase 6 setup
        close = indicator_to_float(
            get_indicator(row, "Close")  # type: ignore[arg-type]
        )
        sma150 = indicator_to_float(
            get_indicator(row, "sma150")  # type: ignore[arg-type]
        )
        drop3d = indicator_to_float(
            get_indicator(row, "drop3d")  # type: ignore[arg-type]
        )

        if math.isnan(close) or math.isnan(sma150) or math.isnan(drop3d):
            return False

        # セットアップ: 終値が150日SMAを上回る & 過去3日で12.5%以上下落
        return (close > sma150) and (drop3d >= 0.125)
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


def system5_setup_predicate(
    row: pd.Series, *, atr_pct_threshold: float | None = None
) -> bool:
    try:
        close = _to_float(row.get("Close"))
        adx7 = _to_float(row.get("adx7"))
        atr_pct = _to_float(row.get("atr_pct"))
        threshold = (
            atr_pct_threshold
            if atr_pct_threshold is not None
            else DEFAULT_ATR_PCT_THRESHOLD
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


# --- System6 (Short momentum) ------------------------------------------------
# 条件: return_6d > 0.20 and uptwodays == True
def system6_setup_predicate(row: pd.Series) -> bool:
    try:
        return_6d = _to_float(row.get("return_6d"))
        uptwo = bool(row.get("uptwodays") or row.get("UpTwoDays"))
        if math.isnan(return_6d):
            return False
        return (return_6d > 0.20) and uptwo
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
    "6": system6_setup_predicate,
    "System6": system6_setup_predicate,
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
    "system6_setup_predicate",
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

    環境変数 VALIDATE_SETUP_PREDICATE が有効でない場合は何もしない。
    速度影響を避けるため最大 sample_max 行までランダム抽出。

    ログ形式 (不一致時):
        [SystemX] setup predicate mismatch: mismatches=N sample=[SYM1,SYM2,...]
    """
    # 型安全な環境変数アクセスに統一
    try:
        from config.environment import get_env_config  # 遅延 import で循環を避ける

        env = get_env_config()
        if not getattr(env, "validate_setup_predicate", False):
            return
    except Exception:
        # 環境設定取得に失敗しても、従来互換のフォールバックで制御
        if os.environ.get("VALIDATE_SETUP_PREDICATE", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
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
                f"[{system_id}] setup predicate mismatch: "
                f"mismatches={len(mismatches)} "
                f"sample={mismatches}"
            )
        except Exception:
            pass


__all__.append("validate_predicate_equivalence")
