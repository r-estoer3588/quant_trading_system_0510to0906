"""
_recompute_indicators の処理を段階別に診断
"""

import pandas as pd

from common.cache_manager import CacheManager
from config.settings import get_settings


def debug_recompute_step_by_step():
    """_recompute_indicators の各段階を診断"""

    settings = get_settings()
    cache_manager = CacheManager(settings)

    # 元データを取得
    base_and_tail = cache_manager._read_base_and_tail("SPY", 330)

    if base_and_tail is None or base_and_tail.empty:
        print("ERROR: base_and_tail データが取得できませんでした")
        return

    print("=== _read_base_and_tail の結果 ===")
    print(f"行数: {len(base_and_tail)}")
    print(f"日付カラムの型: {base_and_tail['date'].dtype}")
    print("最初の5行:")
    print(base_and_tail[["date", "open", "high", "low", "close", "volume"]].head())
    print(f"カラム名: {list(base_and_tail.columns)}")

    # _recompute_indicators を段階的に実行
    print("\n=== _recompute_indicators の段階別実行 ===")

    df = base_and_tail.copy()
    print(f"ステップ1 - コピー後の日付: {df['date'].head()}")

    # ステップ2: 日付変換
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"ステップ2 - pd.to_datetime後の日付: {df['date'].head()}")
    print(f"ステップ2 - 日付のNaN数: {df['date'].isna().sum()}")

    # ステップ3: dropna
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"ステップ3 - dropna後の行数: {len(df)}")
    print(f"ステップ3 - dropna後の日付: {df['date'].head()}")

    # ステップ4: 数値変換
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"ステップ4 - 数値変換後の日付: {df['date'].head()}")

    # ステップ5: カラム名変換
    case_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df_renamed = df.rename(columns={k: v for k, v in case_map.items() if k in df.columns})
    df_renamed["Date"] = df_renamed["date"]
    print(f"ステップ5 - カラム名変換後の日付: {df_renamed['date'].head()}")
    print(f"ステップ5 - 'Date'カラム: {df_renamed['Date'].head()}")

    print(f"\nカラム名リスト: {list(df_renamed.columns)}")


if __name__ == "__main__":
    debug_recompute_step_by_step()
