"""
_recompute_indicators の詳細なデバッグスクリプト
"""

import pandas as pd

from common.cache_manager import CacheManager, standardize_indicator_columns
from common.indicators_common import add_indicators
from config.settings import get_settings


def debug_recompute_indicators_detailed():
    """_recompute_indicators の各ステップを詳細にデバッグ"""

    settings = get_settings()
    cache_manager = CacheManager(settings)

    # rolling に相当するデータを取得
    base_and_tail = cache_manager._read_base_and_tail("SPY", 330)
    if base_and_tail is None or base_and_tail.empty:
        print("ERROR: base_and_tail データが取得できません")
        return

    print("=== 元データ ===")
    print(f"行数: {len(base_and_tail)}")
    print(f"列数: {len(base_and_tail.columns)}")
    print(f"カラム名: {list(base_and_tail.columns)}")

    # _recompute_indicators のステップを再現
    df = base_and_tail.copy()

    # ステップ1: 前処理
    print("\n=== ステップ1: 前処理 ===")
    if df is None or df.empty or "date" not in df.columns:
        print("ERROR: 前処理チェックで失敗")
        return

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        print("ERROR: 必要なカラムが不足")
        return
    print("前処理チェック: ✅")

    # ステップ2: データクリーニング
    print("\n=== ステップ2: データクリーニング ===")
    base = df.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"クリーニング後行数: {len(base)}")
    print(f"日付の最小値: {base['date'].min()}")
    print(f"日付の最大値: {base['date'].max()}")

    # ステップ3: 数値変換
    print("\n=== ステップ3: 数値変換 ===")
    for col in ("open", "high", "low", "close", "volume"):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")
            nan_count = base[col].isna().sum()
            print(f"{col}: NaN数 {nan_count}")

    # ステップ4: カラム名変換
    print("\n=== ステップ4: カラム名変換 ===")
    case_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    base_renamed = base.rename(columns={k: v for k, v in case_map.items() if k in base.columns})
    base_renamed["Date"] = base_renamed["date"]
    print(f"変換後カラム名: {list(base_renamed.columns)}")

    # ステップ5: 既存指標列の削除
    print("\n=== ステップ5: 既存指標列の削除 ===")
    basic_cols = {
        "date",
        "Date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "raw_close",
    }
    indicator_cols = [col for col in base_renamed.columns if col not in basic_cols]
    print(f"削除対象の指標列 ({len(indicator_cols)}個): {indicator_cols[:10]}")
    if indicator_cols:
        base_renamed = base_renamed.drop(columns=indicator_cols)
    print(f"削除後カラム数: {len(base_renamed.columns)}")

    # ステップ6: add_indicators 実行
    print("\n=== ステップ6: add_indicators 実行 ===")
    try:
        enriched = add_indicators(base_renamed)
        print(f"add_indicators 成功! 結果行数: {len(enriched)}")
        print(f"結果列数: {len(enriched.columns)}")

        # 指標列をチェック
        new_indicator_cols = [col for col in enriched.columns if col not in basic_cols]
        print(f"新しい指標列 ({len(new_indicator_cols)}個): {new_indicator_cols[:10]}")

        # サンプル指標のNaN率をチェック
        for col in ["sma25", "rsi4", "atr10"][:3]:
            if col in enriched.columns:
                nan_rate = enriched[col].isna().mean() * 100
                print(f"  {col}: NaN率 {nan_rate:.1f}%")

    except Exception as e:
        print(f"add_indicators でエラー: {e}")
        import traceback

        traceback.print_exc()
        return

    # ステップ7: 後処理
    print("\n=== ステップ7: 後処理 ===")
    enriched = enriched.drop(columns=["Date"], errors="ignore")
    print("Date列削除: ✅")

    # 指標列を標準化
    enriched = standardize_indicator_columns(enriched)
    print("指標列標準化: ✅")

    # 基本列のみ小文字に変換
    basic_cols_check = {"open", "high", "low", "close", "volume", "date"}
    enriched.columns = [c.lower() if c.lower() in basic_cols_check else c for c in enriched.columns]
    print("基本列小文字化: ✅")

    enriched["date"] = pd.to_datetime(enriched.get("date", base["date"]), errors="coerce")
    print(f"最終日付設定: ✅ (NaN数: {enriched['date'].isna().sum()})")

    # ステップ8: 結合処理
    print("\n=== ステップ8: 結合処理 ===")
    ohlcv = {"date", "open", "high", "low", "close", "volume", "raw_close"}
    ohlcv_cols = [col for col in ohlcv if col in df.columns]
    combined = df[ohlcv_cols].copy()
    print(f"OHLCV列を抽出: {ohlcv_cols}")

    # 指標列を追加
    added_count = 0
    for col, series in enriched.items():
        if col in ohlcv:
            continue
        combined[col] = series
        added_count += 1

    print(f"指標列を追加: {added_count}個")
    print(f"最終結果行数: {len(combined)}")
    print(f"最終結果列数: {len(combined.columns)}")

    # 最終結果の指標サンプルをチェック
    print("\n=== 最終結果の指標チェック ===")
    for col in ["sma25", "rsi4", "atr10"]:
        if col in combined.columns:
            nan_rate = combined[col].isna().mean() * 100
            print(f"{col}: NaN率 {nan_rate:.1f}%")


if __name__ == "__main__":
    debug_recompute_indicators_detailed()
