"""_merge_existing_full 修正のテスト"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

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

print("📊 マージテスト開始")
print(f"  既存データ: {existing_full.shape[0]} 行")
print(f"  新規データ: {new_full.shape[0]} 行")
print(f'  既存の最終日: {existing_full["date"].iloc[-1]}')

# マージ実行
result = _merge_existing_full(new_full, existing_full)

print("\n✅ マージ結果:")
print(f"  結果データ: {result.shape[0]} 行")
print(f"  列数: {result.shape[1]}")
print(f'  最終日: {result["date"].iloc[-1]}')

# 既存データが保持されているか確認
print("\n🔍 データ保持確認:")
print(f'  最終3行の日付: {result["date"].tail(3).tolist()}')

# 新規データが追加されているか
new_row = result[result["date"] == "2025-10-04"]
if not new_row.empty:
    print("\n✅ 新規データ追加成功!")
    print(f'  date: {new_row["date"].values[0]}')
    print(f'  open: {new_row["open"].values[0]}')
    print(f'  close: {new_row["close"].values[0]}')
    print(f'  adjclose: {new_row["adjclose"].values[0]} (NaN is expected for new data)')
else:
    print("\n❌ 新規データが見つかりません!")

# 指標列が保持されているか
if "sma200" in result.columns:
    non_null_sma200 = result["sma200"].notna().sum()
    print(f"\n✅ 指標列保持確認: sma200の非NULL行数 = {non_null_sma200}")
else:
    print("\n❌ sma200列が失われています!")

print("\n最終3行のサンプル:")
print(result.tail(3)[["date", "open", "close", "adjclose", "sma200"]])
