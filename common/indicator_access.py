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


def get_indicator(row: Mapping[str, Any], name: str) -> Any:
    """行(dict/Series)から指標値を取得。未取得なら None。

    name は大小混在可。内部的に小文字化してエイリアスを走査。
    """
    if not row:
        return None
    key = name.lower()
    aliases: Iterable[str] = _ALIAS_MAP.get(key, (name,))
    return get_first(row, *aliases)


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
