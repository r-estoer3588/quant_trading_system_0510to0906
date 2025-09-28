"""
修正後のrolling データを確認するスクリプト
"""

from common.cache_manager import CacheManager
from config.settings import get_settings


def check_fixed_rolling_data():
    """修正後のrolling データをチェック"""

    settings = get_settings()
    cache_manager = CacheManager(settings)
    rolling_data = cache_manager.read("SPY", "rolling")

    if rolling_data is None or rolling_data.empty:
        print("ERROR: Rolling データが取得できませんでした")
        return

    print("=== Rolling データの基本情報 ===")
    print(f"行数: {len(rolling_data)}")
    print(f"列数: {len(rolling_data.columns)}")

    # 重複カラムのチェック
    duplicate_cols = rolling_data.columns[rolling_data.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"重複カラム: {duplicate_cols}")
    else:
        print("重複カラム: なし ✅")

    # 日付の確認
    print("\n=== 日付データ ===")
    print(f"日付カラムの型: {rolling_data['date'].dtype}")
    print(f"日付のNaN数: {rolling_data['date'].isna().sum()}")
    if rolling_data["date"].notna().any():
        print(f"日付範囲: {rolling_data['date'].min()} ～ {rolling_data['date'].max()}")
        print("最初の5行の日付:")
        print(rolling_data["date"].head().tolist())

    # OHLCV データの確認
    print("\n=== OHLCV データ ===")
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        if col in rolling_data.columns:
            nan_rate = rolling_data[col].isna().mean() * 100
            print(f"{col}: NaN率 {nan_rate:.1f}%")

    # 主要指標の確認
    print("\n=== 主要指標 ===")
    indicators = ["sma25", "sma50", "rsi3", "rsi4", "atr10", "adx7"]
    for col in indicators:
        if col in rolling_data.columns:
            nan_rate = rolling_data[col].isna().mean() * 100
            print(f"{col}: NaN率 {nan_rate:.1f}%")

            # 正常な値があれば最後の10個を表示
            if nan_rate < 100:
                print(f"  最後の5個の値: {rolling_data[col].tail().tolist()}")

    # 全指標列のNaN率サマリー
    print("\n=== 全指標のNaN率サマリー ===")
    basic_cols = {"date", "open", "high", "low", "close", "volume", "raw_close"}
    indicator_cols = [col for col in rolling_data.columns if col not in basic_cols]

    good_indicators = []
    bad_indicators = []

    for col in indicator_cols:
        nan_rate = rolling_data[col].isna().mean() * 100
        if nan_rate < 50:  # 50%未満のNaN率なら良好
            good_indicators.append(f"{col}({nan_rate:.1f}%)")
        else:
            bad_indicators.append(f"{col}({nan_rate:.1f}%)")

    print(f"良好な指標 ({len(good_indicators)}個): {', '.join(good_indicators[:10])}")
    if len(good_indicators) > 10:
        print(f"  ...他{len(good_indicators) - 10}個")

    if bad_indicators:
        print(f"問題のある指標 ({len(bad_indicators)}個): {', '.join(bad_indicators[:5])}")
        if len(bad_indicators) > 5:
            print(f"  ...他{len(bad_indicators) - 5}個")


if __name__ == "__main__":
    check_fixed_rolling_data()
