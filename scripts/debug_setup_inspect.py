#!/usr/bin/env python
"""Setup判定で0件になる原因を調査するための簡易インスペクタ。

使い方:
  python scripts/debug_setup_inspect.py --symbols SPY AAPL MSFT --limit 1

挙動:
  - CacheManager 経由で rolling を読み込む
  - 末尾行の主要指標値を取得し表示
  - indicator_access.get_indicator で alias 解決した値と raw 列存在状況を並べる

安全:
  - 書き込みなし / ネットワークなし
"""

from __future__ import annotations

import argparse
from typing import Iterable

import pandas as pd

from common.cache_manager import CacheManager
from common.indicator_access import get_indicator, to_float
from config.settings import get_settings

TARGET_COLUMNS = [
    # price trend
    "Close",
    "SMA25",
    "SMA50",
    "SMA100",
    "SMA150",
    "SMA200",
    # momentum / volatility
    "RSI3",
    "RSI4",
    "ADX7",
    "ATR10",
    "ATR20",
    "ATR_Pct",
    "ATR_Ratio",
    # custom
    "Drop3D",
    "drop3d",
    "RETURN_6D",
    "UpTwoDays",
    "TwoDayUp",
]

ALIAS_TEST = [
    ("sma25", "SMA25"),
    ("sma50", "SMA50"),
    ("sma100", "SMA100"),
    ("sma150", "SMA150"),
    ("sma200", "SMA200"),
    ("rsi3", "RSI3"),
    ("rsi4", "RSI4"),
    ("adx7", "ADX7"),
    ("atr10", "ATR10"),
    ("atr_pct", "ATR_Pct"),
    ("drop3d", "Drop3D"),
    ("return_6d", "RETURN_6D"),
    ("twodayup", "TwoDayUp"),
]


def load_last_row(cm: CacheManager, symbol: str) -> pd.Series | None:
    try:
        df = cm.read(symbol, "rolling")
        if df is None or getattr(df, "empty", True):
            return None
        # 正規化: Date / date があれば末尾ソート
        if "date" in df.columns:
            df = df.sort_values("date")
        elif "Date" in df.columns:
            df = df.sort_values("Date")
        return df.iloc[-1]
    except Exception:
        return None


def inspect_symbol(cm: CacheManager, symbol: str) -> dict:
    row = load_last_row(cm, symbol)
    out: dict = {"symbol": symbol, "exists": row is not None}
    if row is None:
        return out
    # raw column presence
    for col in TARGET_COLUMNS:
        out[f"has:{col}"] = col in row.index
    # alias resolution values
    for key, disp in ALIAS_TEST:
        val = get_indicator(row.to_dict(), key)
        out[f"val:{key}"] = to_float(val)
    return out


def main(symbols: Iterable[str], limit: int | None = None):
    settings = get_settings(create_dirs=False)
    cm = CacheManager(settings)
    results = []
    count = 0
    for sym in symbols:
        results.append(inspect_symbol(cm, sym))
        count += 1
        if limit and count >= limit:
            break
    # 表示: カラム縮小
    if not results:
        print("No symbols inspected.")
        return
    df = pd.DataFrame(results)
    # NaN を空文字に変換せずそのまま出す（数値かどうか判断しやすい）
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(df.T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=["SPY"], help="優先的に確認するシンボルリスト")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    main(args.symbols, args.limit)
