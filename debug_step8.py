"""
ステップ8の結合処理を詳しく調査
"""

from common.cache_manager import CacheManager
from config.settings import get_settings


def debug_step8_combination():
    """ステップ8の結合処理を詳細に調査"""

    settings = get_settings()
    cache_manager = CacheManager(settings)

    # テスト用に少量のデータで _recompute_indicators を実行
    base_and_tail = cache_manager._read_base_and_tail("SPY", 50)

    if base_and_tail is None or base_and_tail.empty:
        print("ERROR: base_and_tail データが取得できません")
        return

    print("=== 元データ ===")
    print(f"行数: {len(base_and_tail)}")
    print("最初の3行のclose値:")
    print(base_and_tail["close"].head(3).tolist())

    # _recompute_indicators を実行
    result = cache_manager._recompute_indicators(base_and_tail)

    print("\n=== _recompute_indicators 結果 ===")
    print(f"結果行数: {len(result)}")
    print(f"結果列数: {len(result.columns)}")

    # close列の確認
    print("結果のclose値 (最初の3行):")
    print(result["close"].head(3).tolist())

    # 指標列の確認
    basic_cols = {"date", "open", "high", "low", "close", "volume", "raw_close"}
    indicator_cols = [col for col in result.columns if col not in basic_cols]
    print(f"指標列 ({len(indicator_cols)}個): {indicator_cols[:5]}")

    # サンプル指標の値とNaN率を確認
    for col in ["sma25", "rsi4", "atr10"]:
        if col in result.columns:
            values = result[col]
            nan_rate = values.isna().mean() * 100
            print(f"\n{col}:")
            print(f"  NaN率: {nan_rate:.1f}%")
            print(f"  型: {values.dtype}")
            print(f"  最初の5個: {values.head().tolist()}")
            print(f"  最後の5個: {values.tail().tolist()}")

            # 有効な値の統計
            valid_values = values.dropna()
            if len(valid_values) > 0:
                print(f"  有効値数: {len(valid_values)}")
                print(f"  平均: {valid_values.mean():.4f}")
                print(f"  最小: {valid_values.min():.4f}")
                print(f"  最大: {valid_values.max():.4f}")


if __name__ == "__main__":
    debug_step8_combination()
