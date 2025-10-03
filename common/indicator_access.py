"""Indicator access helpers.

目的: 異なる大文字/小文字・旧別名を統一的に解決して値を取得する軽量ユーティリティ。
過剰な抽象化を避け、現在確認された揺れのみ対応。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

# 別名マッピング（小文字キーで統一）
_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "sma25": ("sma25", "SMA25"),
    "sma50": ("sma50", "SMA50"),
    "sma100": ("sma100", "SMA100"),
    "sma150": ("sma150", "SMA150"),
    "sma200": ("sma200", "SMA200"),
    "rsi3": ("rsi3", "RSI3"),
    "rsi4": ("rsi4", "RSI4"),
    "adx7": ("adx7", "ADX7"),
    "atr10": ("atr10", "ATR10"),
    "atr_pct": ("atr_pct", "ATR_Pct", "ATR_PCT", "ATR_Ratio", "atr_ratio"),
    "drop3d": ("drop3d", "Drop3D"),
    "return_6d": ("return_6d", "RETURN_6D"),
    "twodayup": ("twodayup", "TwoDayUp"),
    "uptwodays": ("uptwodays", "UpTwoDays", "TwoDayUp", "twodayup"),
}


def get_first(mapping: Mapping[str, Any], *candidates: str) -> Any:
    for c in candidates:
        if c in mapping:
            return mapping[c]
    return None


def _build_lower_map(row: Mapping[str, Any]) -> dict[str, str]:
    """row のキーを小文字化 -> 元キー へのマップ生成 (必要時のみ)。"""
    lm: dict[str, str] = {}
    try:
        for k in row.keys():  # type: ignore[attr-defined]
            if isinstance(k, str):
                lk = k.lower()
                if lk not in lm:
                    lm[lk] = k
    except Exception:
        pass
    return lm


def get_indicator(row: Mapping[str, Any], name: str) -> Any:
    """行(dict/Series)から指標値を取得。未取得なら None。

    仕様追加: エイリアス探索で見つからない場合、全キーの lowercase マップを一度構築し
    name.lower() と一致する元キーがあればその値を返す（列が全て小文字化されているケースの救済）。

    修正: pandas.Series に対し ``if not row`` は ValueError (真偽値が不明) を発生させるため
    None/空判定は is None と len==0 のみに限定。
    """
    if row is None:  # 明示的 None
        return None
    try:
        # pandas Series / dict いずれでも len 0 を空扱い
        if hasattr(row, "__len__") and len(row) == 0:  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    key = name.lower()
    aliases: Iterable[str] = _ALIAS_MAP.get(key, (name,))
    val = get_first(row, *aliases)
    if val is not None:
        return val
    # フォールバック lowercase 探索
    lower_map = _build_lower_map(row)
    orig = lower_map.get(key)
    if orig is not None:
        try:
            return row[orig]  # type: ignore[index]
        except Exception:
            return None
    return None


def to_float(val: Any) -> float:
    try:
        if val is None:
            return math.nan
        f = float(val)
        if math.isnan(f):
            return math.nan
        return f
    except Exception:
        return math.nan


def is_true(val: Any) -> bool:
    return bool(val)


__all__ = ["get_indicator", "to_float", "is_true"]
