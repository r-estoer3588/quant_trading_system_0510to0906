"""compute_base_indicators のテスト"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from common.cache_manager import compute_base_indicators
from scripts.update_from_bulk_last_day import _merge_existing_full

# テストデータ
new_data = {
    "date": ["2025-10-04"],
    "open": [142.0],
    "high": [143.0],
    "low": [141.0],
    "close": [142.5],
    "volume": [1000000],
}
new_full = pd.DataFrame(new_data)
existing_full = pd.read_csv("data_cache/full_backup/A.csv")

print("📊 指標計算テスト開始")

# マージ実行
merged = _merge_existing_full(new_full, existing_full)
print(f"  マージ後データ: {merged.shape[0]} 行")

# 指標計算
print("\n🔧 compute_base_indicators を実行中...")
try:
    result = compute_base_indicators(merged)
    print(f"✅ 成功! 結果: {result.shape[0]} 行, {result.shape[1]} 列")

    # 新規データの指標を確認
    if "Date" in result.columns:
        result_indexed = result.set_index("Date")
        new_row = result_indexed.loc["2025-10-04"]
        print("\n新規行の指標計算結果:")
        print(f'  SMA200: {new_row.get("SMA200", "N/A")}')
        print(f'  RSI4: {new_row.get("RSI4", "N/A")}')
        print(f'  ATR20: {new_row.get("ATR20", "N/A")}')
    else:
        print("\n⚠️ Date列が見つかりません")
        print(f"列: {list(result.columns)[:10]}...")

except Exception as e:
    print(f"❌ エラー: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
