from __future__ import annotations

from typing import Callable, Dict, Optional

import pandas as pd

from indicators_common import add_indicators


def _ensure_price_columns_upper(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # 既に大文字があれば尊重し、無い場合のみ小文字から補完
    if "Open" not in x.columns and "open" in x.columns:
        x["Open"] = x["open"]
    if "High" not in x.columns and "high" in x.columns:
        x["High"] = x["high"]
    if "Low" not in x.columns and "low" in x.columns:
        x["Low"] = x["low"]
    if "Close" not in x.columns and "close" in x.columns:
        x["Close"] = x["close"]
    if "Volume" not in x.columns and "volume" in x.columns:
        x["Volume"] = x["volume"]
    return x


def precompute_shared_indicators(
    basic_data: Dict[str, pd.DataFrame],
    *,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    basic_data の各 DataFrame に共有インジケータ列を付与して返す。

    - 入力はローリング/ベース由来の最小カラム（小文字）でも可。
    - 価格系カラム（Open/High/Low/Close/Volume）は大文字を補完してから計算。
    - 出力は元の DataFrame に `add_indicators` で追加された列を結合。
    - 既存列は上書きしない方針（同名が存在すればそのまま残す）。
    """
    if not basic_data:
        return basic_data
    out: Dict[str, pd.DataFrame] = {}
    total = len(basic_data)
    for idx, (sym, df) in enumerate(basic_data.items(), start=1):
        try:
            if df is None or getattr(df, "empty", True):
                out[sym] = df
                continue
            # 指標計算用に大文字カラムを補完
            work = _ensure_price_columns_upper(df)
            # 計算（`add_indicators` は安全に不足時は NaN を入れる）
            ind_df = add_indicators(work)
            # 新規列のみを元 df に結合（既存カラムは保持）
            new_cols = [c for c in ind_df.columns if c not in df.columns]
            if new_cols:
                merged = df.copy()
                for c in new_cols:
                    merged[c] = ind_df[c]
                out[sym] = merged
            else:
                out[sym] = df
        except Exception:
            # 失敗時はそのまま返す（堅牢性重視）
            out[sym] = df
        if log and (idx % 1000 == 0 or idx == total):
            try:
                log(f"🧮 共有指標 前計算: {idx}/{total}")
            except Exception:
                pass
    return out


__all__ = ["precompute_shared_indicators"]
