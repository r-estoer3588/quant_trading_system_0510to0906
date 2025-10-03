"""
Rolling データの全指標を詳細に確認するスクリプト
"""

from common.cache_manager import CacheManager
from config.settings import get_settings


def check_all_indicators():
    """Rolling データの全指標を詳細に確認"""

    settings = get_settings()
    cache_manager = CacheManager(settings)
    rolling_data = cache_manager.read("SPY", "rolling")

    if rolling_data is None or rolling_data.empty:
        print("ERROR: Rolling データが取得できませんでした")
        return

    print("=== Rolling SPY データの全容 ===")
    print(f"行数: {len(rolling_data)}")
    print(f"列数: {len(rolling_data.columns)}")

    # 全カラムを分類
    basic_cols = {"date", "open", "high", "low", "close", "volume", "raw_close"}
    all_cols = set(rolling_data.columns)
    indicator_cols = sorted(all_cols - basic_cols)

    print(f"\n=== 基本データ列 ({len(basic_cols)}個) ===")
    for col in sorted(basic_cols):
        if col in rolling_data.columns:
            print(f"  {col}")

    print(f"\n=== 全指標列 ({len(indicator_cols)}個) ===")

    # 指標をカテゴリ別に分類
    categories = {
        "ATR": [col for col in indicator_cols if col.lower().startswith("atr")],
        "SMA": [col for col in indicator_cols if col.lower().startswith("sma")],
        "RSI": [col for col in indicator_cols if col.lower().startswith("rsi")],
        "ADX": [col for col in indicator_cols if col.lower().startswith("adx")],
        "ROC": [col for col in indicator_cols if col.lower().startswith("roc")],
        "Dollar Volume": [col for col in indicator_cols if "dollarvolume" in col.lower()],
        "Volume": [col for col in indicator_cols if "avgvolume" in col.lower()],
        "Return": [col for col in indicator_cols if "return" in col.lower()],
        "Volatility": [
            col for col in indicator_cols if col.lower() in ["hv50", "atr_pct", "atr_ratio"]
        ],
        "Price Pattern": [
            col for col in indicator_cols if col.lower() in ["uptwodays", "twodayup", "drop3d"]
        ],
        "Min/Max": [col for col in indicator_cols if col.lower().startswith(("min_", "max_"))],
        "Other": [],
    }

    # 分類されなかった指標をOtherに追加
    categorized = set()
    for cat_indicators in categories.values():
        categorized.update(cat_indicators)
    categories["Other"] = [col for col in indicator_cols if col not in categorized]

    # カテゴリ別表示と統計
    total_good = 0
    total_problematic = 0

    for category, indicators in categories.items():
        if not indicators:
            continue

        print(f"\n--- {category} ({len(indicators)}個) ---")

        category_good = 0
        category_problematic = 0

        for col in sorted(indicators):
            if col in rolling_data.columns:
                nan_rate = rolling_data[col].isna().mean() * 100
                status = "✅" if nan_rate < 50 else "⚠️"
                print(f"  {status} {col}: NaN率 {nan_rate:.1f}%")

                if nan_rate < 50:
                    category_good += 1
                    total_good += 1
                else:
                    category_problematic += 1
                    total_problematic += 1

                # 良好な指標の場合、サンプル値も表示
                if nan_rate < 10 and len(rolling_data) > 0:
                    last_values = rolling_data[col].dropna().tail(3).tolist()
                    if last_values:
                        print(f"      最新3値: {[f'{v:.3f}' for v in last_values]}")

        print(f"  カテゴリ内良好: {category_good}個, 問題: {category_problematic}個")

    print("\n=== 総計 ===")
    print(f"全指標数: {len(indicator_cols)}個")
    print(f"良好な指標 (NaN率<50%): {total_good}個")
    print(f"問題のある指標 (NaN率≥50%): {total_problematic}個")
    print(f"指標正常率: {total_good / len(indicator_cols) * 100:.1f}%")

    # 最高・最低NaN率の指標
    best_indicators = []
    worst_indicators = []

    for col in indicator_cols:
        if col in rolling_data.columns:
            nan_rate = rolling_data[col].isna().mean() * 100
            if nan_rate == 0:
                best_indicators.append(col)
            elif nan_rate >= 50:
                worst_indicators.append((col, nan_rate))

    if best_indicators:
        print(f"\n完璧な指標 (NaN率0%): {len(best_indicators)}個")
        print(f"  {', '.join(sorted(best_indicators)[:10])}")
        if len(best_indicators) > 10:
            print(f"  ...他{len(best_indicators)-10}個")

    if worst_indicators:
        print(f"\n問題のある指標: {len(worst_indicators)}個")
        for col, rate in sorted(worst_indicators, key=lambda x: x[1], reverse=True):
            print(f"  {col}: {rate:.1f}%")


if __name__ == "__main__":
    check_all_indicators()
